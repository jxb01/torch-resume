"""执行迁移：把旧权重与旧优化器状态搬到新模型上。

两个容易漏、漏了必然出问题的点：
  1. 新模型是构造函数刚建好的，所以 "reset" 意味着**什么都不做**（用新模型的初始化）
  2. **优化器状态也要迁**，尤其是 Adam 的动量——只搬权重会导致续训剧烈震荡
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .plan import MigrationPlan

import torch

from .plan import MigrationPlan, diff
from .state import name_map, opt_state_by_name


@dataclass
class MigrateStats:
    kept: int = 0
    partial: int = 0
    renamed: int = 0
    reset: int = 0
    dropped: int = 0
    opt_migrated: int = 0
    opt_reset: int = 0
    # 本次用的迁移计划。分组和 warmup 需要它来判断每个参数的类别。
    plan: Optional[MigrationPlan] = None

    def report(self) -> str:
        return ("权重: 保留=%d 部分继承=%d 重命名=%d 重置=%d 删除=%d\n"
                "优化器状态: 迁移=%d 重置=%d") % (
            self.kept, self.partial, self.renamed, self.reset, self.dropped,
            self.opt_migrated, self.opt_reset)


def apply_plan(
    new_model: torch.nn.Module, plan: MigrationPlan, old_state: Dict[str, torch.Tensor]
) -> MigrateStats:
    """把旧 state_dict 按计划搬进新模型。原地拷贝，不破坏优化器持有的参数引用。"""
    st = MigrateStats()
    target = dict(new_model.named_parameters())
    target.update(dict(new_model.named_buffers()))

    with torch.no_grad():
        for a in plan.actions:
            if a.kind == "keep":
                target[a.name].copy_(old_state[a.name])
                st.kept += 1
            elif a.kind == "partial":
                target[a.name][a.slices].copy_(old_state[a.source])
                st.partial += 1
            elif a.kind == "rename":
                target[a.name].copy_(old_state[a.source])
                st.renamed += 1
            elif a.kind == "reset":
                st.reset += 1          # 新模型自己初始化过了，什么都不做
            elif a.kind == "drop":
                st.dropped += 1
    return st


def _copy_opt_tensor(
    old: torch.Tensor, ref: torch.Tensor, slices=None
) -> Optional[torch.Tensor]:
    """把旧优化器状态的张量搬到新形状上。

    注意：切片以外的区域必须是 **0**，不能用别的东西填。
    exp_avg_sq 是非负量，一旦被填进负值（比如参数本身），开根号就变 NaN。
    """
    if not torch.is_tensor(old):
        return None
    if slices is not None:
        if old.ndim != ref.ndim:
            return None
        out = torch.zeros_like(ref)
        out[slices] = old[slices].to(ref.device, ref.dtype)
        return out
    if tuple(old.shape) != tuple(ref.shape):
        return None
    return old.to(ref.device, ref.dtype).clone()


def migrate_optimizer(
    new_model: torch.nn.Module,
    new_optimizer: torch.optim.Optimizer,
    old_opt_state: Dict[str, Dict[str, Any]],
    plan: MigrationPlan,
) -> MigrateStats:
    """按名字把优化器状态搬过去；partial 的参数用同样的切片继承动量。"""
    st = MigrateStats()
    by_name = name_map(new_model)
    plan_by_name = {a.name: a for a in plan.actions}

    for group in new_optimizer.param_groups:
        for p in group["params"]:
            name = next((n for n, q in by_name.items() if q is p), None)
            if name is None:
                continue
            a = plan_by_name.get(name)
            if a is None or a.kind in ("reset", "drop"):
                st.opt_reset += 1
                continue
            src_name = a.source if a.kind in ("partial", "rename") else name
            old = old_opt_state.get(src_name)
            if not old:
                st.opt_reset += 1
                continue
            new_state: Dict[str, Any] = {}
            for k, v in old.items():
                if torch.is_tensor(v) and v.ndim > 0:
                    copied = _copy_opt_tensor(v, p, a.slices if a.kind == "partial" else None)
                    new_state[k] = copied if copied is not None else v.detach().clone()
                elif torch.is_tensor(v):
                    new_state[k] = v.detach().clone()
                else:
                    new_state[k] = v
            new_optimizer.state[p] = new_state
            st.opt_migrated += 1
    return st


def migrate(
    new_model: torch.nn.Module,
    new_optimizer: torch.optim.Optimizer,
    old_model: torch.nn.Module,
    old_optimizer: torch.optim.Optimizer,
    *,
    allow_rename: bool = False,
    partial_min_ratio: float = 0.5,
    verbose: bool = True,
) -> MigrateStats:
    """一步到位：对比结构 -> 搬权重 -> 搬优化器状态。"""
    old_state = {k: v.detach().cpu() for k, v in old_model.state_dict().items()}
    plan = diff(new_model, old_state, allow_rename=allow_rename,
                partial_min_ratio=partial_min_ratio)
    if verbose:
        print(plan.report())
    # 必须用计划里那一份预处理过的旧权重（键已按手术记录重命名/复制）
    st = apply_plan(new_model, plan, plan.old_state if plan.old_state is not None
                    else old_state)
    st2 = migrate_optimizer(
        new_model, new_optimizer, opt_state_by_name(old_model, old_optimizer), plan)
    st.opt_migrated, st.opt_reset = st2.opt_migrated, st2.opt_reset
    st.plan = plan
    if verbose:
        print("-" * 66)
        print(st.report())
    return st
