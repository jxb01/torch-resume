"""优化器状态失配检测。

换 loss 或改结构之后，Adam 的二阶矩 exp_avg_sq 是"旧世界的统计量"。
beta2 = 0.999 时它的记忆约 1/(1-beta2) = 1000 步，期间会给出错误的每参数有效步长：
  v 偏小 -> 有效步长偏大 -> 震荡
  v 偏大 -> 有效步长偏小 -> 停滞（最阴险，参数看起来在动，实际没动）

做法：恢复前先用新 loss 在暂停点**试算一次**，逐参数比对，再决定动谁。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch


@dataclass
class MismatchRow:
    name: str
    v_ref: float        # mean(sqrt(exp_avg_sq))
    g_ref: float        # 新梯度的 rms
    ratio: float        # v_ref / g_ref
    verdict: str
    g_sq: Optional[torch.Tensor] = None    # 逐元素的 E[g^2]，用于接管


@dataclass
class MismatchReport:
    rows: List[MismatchRow] = field(default_factory=list)
    threshold: float = 4.0
    n_batches: int = 0
    # 参数对象 id -> 名字。放在这里而不是塞进优化器 state，
    # 否则会污染 optimizer.state_dict()，load_state_dict 时可能出错。
    param_names: Dict[int, str] = field(default_factory=dict)

    @property
    def n_stall(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "stall")

    @property
    def n_oscillate(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "oscillate")

    @property
    def n_ok(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "ok")

    @property
    def n_invalid(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "invalid")

    def report(self, top: int = 12) -> str:
        lines = ["优化器状态失配检测", "=" * 66]
        lines.append("  阈值 %.1fx，试算 %d 个 batch" % (self.threshold, self.n_batches))
        bad = [r for r in self.rows if r.verdict != "ok"]
        bad.sort(key=lambda r: -abs(torch.log(torch.tensor(max(r.ratio, 1e-12)))))
        for r in bad[:top]:
            if r.verdict == "stall":
                note = "v 偏大 -> 有效步长偏小（会停滞，难发现）"
            elif r.verdict == "invalid":
                note = "状态非有限（NaN/Inf 或梯度为 0）-> 必须重置"
            else:
                note = "v 偏小 -> 有效步长偏大（会震荡）"
            lines.append("  %-30s ratio=%8.3f  %s" % (r.name, r.ratio, note))
        if len(bad) > top:
            lines.append("  ... 另有 %d 个" % (len(bad) - top))
        lines.append("-" * 66)
        lines.append("  汇总: 停滞=%d  震荡=%d  非有限=%d  正常=%d  共=%d" % (
            self.n_stall, self.n_oscillate, self.n_invalid, self.n_ok, len(self.rows)))
        if self.n_stall or self.n_oscillate or self.n_invalid:
            lines.append("  建议: 对失配参数重置 exp_avg_sq，并加 warmup")
        return "\n".join(lines)

    def apply(self, optimizer: torch.optim.Optimizer, *, policy: str = "threshold") -> int:
        """把检测结果落到优化器上。

        policy:
          none      - 只报告
          threshold - 只动失配超阈值的参数（默认，最保守）
          all       - 全部重置
        重置方式：把 exp_avg_sq 设为试算测得的 E[g^2]（"接管"，而不是清零重来）。
        """
        if policy == "none":
            return 0
        rows = {r.name: r for r in self.rows}
        n = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                st = optimizer.state.get(p)
                if not st or "exp_avg_sq" not in st:
                    continue
                name = self.param_names.get(id(p))
                r = rows.get(name) if name else None
                if r is None:
                    # 退化路径：按张量形状反查（同形状多个参数时可能不精确）
                    r = next((x for x in self.rows
                              if x.g_sq is not None and x.g_sq.shape == p.shape
                              and x.verdict != "ok"), None)
                if r is None or r.g_sq is None:
                    continue
                if policy == "threshold" and r.verdict == "ok":
                    continue
                st["exp_avg_sq"] = r.g_sq.to(p.device, st["exp_avg_sq"].dtype).clone()
                if "exp_avg" in st and torch.is_tensor(st["exp_avg"]):
                    if not torch.isfinite(st["exp_avg"]).all():
                        st["exp_avg"] = torch.nan_to_num(st["exp_avg"], nan=0.0)
                n += 1
        return n


def detect_mismatch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
    batches: Iterable[Any],
    *,
    threshold: float = 4.0,
    restore_grads: bool = True,
    max_batches: int = 8,
) -> MismatchReport:
    """在暂停点用新 loss 试算，逐参数比对 sqrt(exp_avg_sq) 与新梯度 rms。"""
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    saved = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in params}
    by_id = {id(p): n for n, p in params}

    model.zero_grad(set_to_none=True)
    acc = {n: torch.zeros(p.shape, dtype=torch.float32, device=p.device) for n, p in params}
    count = 0
    for b in batches:
        if count >= max_batches:
            break
        loss = loss_fn(model, b)
        loss.backward()
        with torch.no_grad():
            for n, p in params:
                if p.grad is not None:
                    acc[n] += p.grad.detach().float().pow(2)
        model.zero_grad(set_to_none=True)
        count += 1

    rep = MismatchReport(threshold=threshold, n_batches=count)
    rep.param_names = {id(p): n for n, p in params}

    for n, p in params:
        st = optimizer.state.get(p)
        if not st or "exp_avg_sq" not in st:
            continue
        v = st["exp_avg_sq"].detach().float()
        v_ref = float(v.sqrt().mean().item())
        g_sq = acc[n] / max(count, 1)
        g_ref = float(g_sq.mean().sqrt().item())
        ratio = (v_ref / g_ref) if g_ref > 0 else float("inf")
        # 非有限值绝不能被当成 ok 静默放过 —— 那正是最该处理的情形
        if not (math.isfinite(v_ref) and math.isfinite(g_ref)) or g_ref <= 0:
            verdict = "invalid"
        elif ratio > threshold:
            verdict = "stall"
        elif ratio < 1.0 / threshold:
            verdict = "oscillate"
        else:
            verdict = "ok"
        rep.rows.append(MismatchRow(n, v_ref, g_ref, ratio, verdict,
                                    g_sq.to(p.device)))

    if restore_grads:
        with torch.no_grad():
            for n, p in params:
                g = saved[n]
                p.grad = g.to(p.device) if g is not None else None
    return rep
