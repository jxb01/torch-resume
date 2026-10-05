"""参数分组与自动 warmup。

迁移之后参数天然分成几类，它们该用不同的学习率、不同的预热长度：

    完全继承     照旧
    部分继承     照旧（已训练的部分仍然有效）
    全新         权重是随机的 -> 通常给更高 lr，但要预热，否则早期梯度会冲坏
    失配         优化器状态被重置 -> 必须预热，让二阶矩重新积累

以前的版本只"建议加 warmup"，现在把建议变成执行。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .plan import MigrationPlan

KIND_KEPT = "kept"
KIND_PARTIAL = "partial"
KIND_FRESH = "fresh"
KIND_MISMATCHED = "mismatched"

KIND_ORDER = (KIND_FRESH, KIND_MISMATCHED, KIND_PARTIAL, KIND_KEPT)

KIND_LABEL = {
    KIND_FRESH: "全新（权重从零初始化）",
    KIND_MISMATCHED: "优化器状态失配",
    KIND_PARTIAL: "部分继承",
    KIND_KEPT: "完全继承",
}

# 默认给全新参数更高的 lr —— 它们需要更快地追上来。
# 失配参数不动 lr，靠 warmup 过渡。
DEFAULT_LR_FACTOR = {
    KIND_KEPT: 1.0,
    KIND_PARTIAL: 1.0,
    KIND_FRESH: 2.0,
    KIND_MISMATCHED: 1.0,
}


def classify(
    model: torch.nn.Module,
    plan: MigrationPlan,
    mismatch=None,
) -> Dict[str, str]:
    """参数名 -> 类别。

    一个参数可能同时"部分继承"且"优化器状态失配"，取更需要处理的那个：
    全新 > 失配 > 部分继承 > 完全继承。
    """
    names = set(dict(model.named_parameters()).keys())
    kind: Dict[str, str] = {}
    for a in plan.actions:
        if a.name not in names:
            continue
        if a.kind == "reset":
            kind[a.name] = KIND_FRESH
        elif a.kind == "partial":
            kind[a.name] = KIND_PARTIAL
        elif a.kind in ("keep", "rename"):
            kind[a.name] = KIND_KEPT

    if mismatch is not None:
        for r in mismatch.rows:
            if r.verdict == "ok" or kind.get(r.name) == KIND_FRESH:
                continue
            kind[r.name] = KIND_MISMATCHED

    for n in names:
        kind.setdefault(n, KIND_KEPT)
    return kind


def retag_groups(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    plan: MigrationPlan,
    *,
    mismatch=None,
    lr_by_kind: Optional[Dict[str, float]] = None,
    base_lr: Optional[float] = None,
) -> List[dict]:
    """按类别重新划分 optimizer.param_groups。

    **optimizer.state 不受影响**（它按参数对象索引），所以调用后已迁移的动量还在。
    每个新 group 上会带一个 plan_kind 标签，供 PlanWarmup 读取。
    """
    kind = classify(model, plan, mismatch)
    name_of = {id(p): n for n, p in model.named_parameters()}

    base: Dict[str, object] = dict(optimizer.defaults)
    for g in optimizer.param_groups:
        base.update({k: v for k, v in g.items() if k != "params"})
    if base_lr is None:
        base_lr = float(base.get("lr", 1e-3))

    factors = dict(DEFAULT_LR_FACTOR)
    if lr_by_kind:
        factors.update(lr_by_kind)

    buckets: Dict[str, List[torch.nn.Parameter]] = {}
    for g in optimizer.param_groups:
        for p in g["params"]:
            k = kind.get(name_of.get(id(p), ""), KIND_KEPT)
            buckets.setdefault(k, []).append(p)

    new_groups: List[dict] = []
    for k in KIND_ORDER:
        if not buckets.get(k):
            continue
        g = {key: val for key, val in base.items() if key != "params"}
        g["params"] = buckets[k]
        g["lr"] = base_lr * float(factors.get(k, 1.0))
        g["plan_kind"] = k
        new_groups.append(g)

    if new_groups:
        optimizer.param_groups[:] = new_groups
    return new_groups


class PlanWarmup(torch.optim.lr_scheduler.LambdaLR):
    """按参数组的 plan_kind 做线性 warmup。

    每个组从 start_factor 倍 lr 起步，在各自指定的步数内线性回到原值。

        sched = tr.PlanWarmup(opt, {"fresh": 500, "mismatched": 200})
        for step in ...:
            ...
            sched.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_by_kind: Dict[str, int],
        *,
        start_factor: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if not 0.0 < start_factor <= 1.0:
            raise ValueError("start_factor 必须在 (0, 1] 之间")
        self._start_factor = start_factor
        steps = [int(warmup_by_kind.get(g.get("plan_kind", KIND_KEPT), 0))
                 for g in optimizer.param_groups]
        self.warmup_steps = steps
        super().__init__(optimizer, [self._lambda(s) for s in steps],
                         last_epoch=last_epoch)

    def _lambda(self, steps: int):
        sf = self._start_factor

        def f(epoch: int) -> float:
            if steps <= 0:
                return 1.0
            t = min(1.0, float(epoch) / float(steps))
            return sf + (1.0 - sf) * t

        return f

    def report(self) -> str:
        lines = ["warmup 计划", "=" * 62]
        for g, s in zip(self.optimizer.param_groups, self.warmup_steps):
            k = g.get("plan_kind", KIND_KEPT)
            lines.append("  %-12s lr=%-10.3e 参数 %-4d 预热 %d 步%s"
                         % (k, g["lr"], len(g["params"]), s,
                            "" if s else "（不预热）"))
        return "\n".join(lines)


def group_report(optimizer: torch.optim.Optimizer) -> str:
    """列出当前参数分组。"""
    lines = ["参数分组", "=" * 62]
    for g in optimizer.param_groups:
        k = g.get("plan_kind", KIND_KEPT)
        lines.append("  %-12s lr=%-10.3e 参数 %-4d  %s"
                     % (k, g.get("lr", float("nan")), len(g["params"]),
                        KIND_LABEL.get(k, "")))
    return "\n".join(lines)
