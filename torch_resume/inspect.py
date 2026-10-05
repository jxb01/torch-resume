"""暂停、查看、改权重、改结构、继承续训。

这是这个库唯一别人没有的东西：训练到一半改结构或改 loss，不用从头再来。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from .migrate import MigrateStats, migrate
from .mismatch import MismatchReport, detect_mismatch


# ---------------------------------------------------------------- 查看

@dataclass
class ParamStat:
    name: str
    shape: tuple
    dtype: str
    mean: float
    std: float
    absmax: float
    nan: int
    frozen: bool

    def line(self) -> str:
        return ("  %-34s %-18s %-8s mean %+9.5f  std %8.5f  |max| %8.5f  nan %d%s"
                % (self.name, str(tuple(self.shape)), self.dtype,
                   self.mean, self.std, self.absmax, self.nan,
                   "  [冻结]" if self.frozen else ""))


def param_table(model: torch.nn.Module, pattern: Optional[str] = None) -> List[ParamStat]:
    """逐参数统计。零维张量直接取值，不需要 .item() 之外的东西。"""
    out: List[ParamStat] = []
    for name, p in model.named_parameters():
        if pattern and not re.search(pattern, name):
            continue
        d = p.detach().float()
        out.append(ParamStat(
            name=name,
            shape=tuple(p.shape),
            dtype=str(p.dtype).replace("torch.", ""),
            mean=float(d.mean().item()) if d.numel() else 0.0,
            std=float(d.std().item()) if d.numel() > 1 else 0.0,
            absmax=float(d.abs().max().item()) if d.numel() else 0.0,
            nan=int(torch.isnan(d).sum().item()),
            frozen=not p.requires_grad,
        ))
    return out


def print_params(model: torch.nn.Module, pattern: Optional[str] = None) -> None:
    rows = param_table(model, pattern)
    if not rows:
        print("  (无匹配参数)")
        return
    for r in rows:
        print(r.line())
    total = sum(1 for _ in model.parameters())
    print("  --- 匹配 %d / 共 %d 个参数 ---" % (len(rows), total))


# ---------------------------------------------------------------- 改

def set_param(model: torch.nn.Module, name: str, value: Any) -> None:
    """把某个参数整体填成 value（标量广播，或同形状张量）。"""
    target = dict(model.named_parameters()).get(name)
    if target is None:
        raise KeyError("没有这个参数: %s" % name)
    with torch.no_grad():
        if torch.is_tensor(value):
            target.copy_(value.to(target.device, target.dtype))
        else:
            target.fill_(float(value))


def freeze(model: torch.nn.Module, pattern: str, value: bool = True) -> int:
    n = 0
    for name, p in model.named_parameters():
        if re.search(pattern, name):
            p.requires_grad = not value
            n += 1
    return n


# ---------------------------------------------------------------- 续训

def resume_with_edit(
    old_model: torch.nn.Module,
    old_optimizer: torch.optim.Optimizer,
    new_model: torch.nn.Module,
    new_optimizer: torch.optim.Optimizer,
    *,
    loss_fn: Optional[Callable[[torch.nn.Module, Any], torch.Tensor]] = None,
    batches: Optional[Iterable[Any]] = None,
    threshold: float = 4.0,
    policy: str = "threshold",
    allow_rename: bool = False,
    partial_min_ratio: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """改完结构 / 改完 loss 之后，一条命令接着训。

    做三件事：
      1. 结构 diff + 权重继承（未改动的保留，改动的用新模型自己的初始化）
      2. 优化器状态迁移（含 Adam 动量；部分继承的参数按切片迁动量）
      3. 若给了 loss_fn 与 batches，用新 loss 试算一次，做**失配检测**并按策略处理

    Returns: {"migrate": MigrateStats, "mismatch": MismatchReport | None}
    """
    st: MigrateStats = migrate(
        new_model, new_optimizer, old_model, old_optimizer,
        allow_rename=allow_rename, partial_min_ratio=partial_min_ratio,
        verbose=verbose)

    rep: Optional[MismatchReport] = None
    if loss_fn is not None and batches is not None:
        rep = detect_mismatch(new_model, new_optimizer, loss_fn, batches,
                              threshold=threshold)
        if verbose:
            print(rep.report())
        n = rep.apply(new_optimizer, policy=policy)
        if verbose:
            print("[失配处理] 策略=%s，已重置 %d 个参数的 exp_avg_sq" % (policy, n))
    elif verbose:
        print("[失配检测] 跳过（未提供 loss_fn / batches）"
              " —— 换过 loss 的话强烈建议补上")

    return {"migrate": st, "mismatch": rep}


# ---------------------------------------------------------------- 交互式

HELP = """可用命令:
  list [正则]            列出参数统计
  show <正则>            同上（详细）
  set <名字> <数值>      把参数填成数值
  zero <正则>            置零
  freeze <正则>          冻结（不参与梯度）
  unfreeze <正则>        解冻
  nan                    找出含 NaN/Inf 的参数
  help                   显示本帮助
  resume                 退出检查器，继续训练
"""


class Inspector:
    """暂停训练，进去看和改。非交互环境可直接调 run_script()。"""

    def __init__(self, model: torch.nn.Module, *, step: Optional[int] = None,
                 title: str = "训练检查器") -> None:
        self.model = model
        self.step = step
        self.title = title

    def banner(self) -> None:
        head = "%s" % self.title
        if self.step is not None:
            head += "（step %s）" % self.step
        print("=" * 66)
        print(head)
        print("=" * 66)
        print(HELP)

    def execute(self, line: str) -> bool:
        """执行一条命令。返回 False 表示退出。"""
        parts = line.strip().split()
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("resume", "exit", "quit", "q", "done"):
            return False
        if cmd == "help":
            print(HELP)
        elif cmd in ("list", "ls", "show"):
            print_params(self.model, args[0] if args else None)
        elif cmd == "set":
            if len(args) < 2:
                print("  用法: set <名字> <数值>")
            else:
                set_param(self.model, args[0], float(args[1]))
                print("  已设置 %s = %s" % (args[0], args[1]))
        elif cmd == "zero":
            if not args:
                print("  用法: zero <正则>")
            else:
                n = 0
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if re.search(args[0], name):
                            p.zero_()
                            n += 1
                print("  已置零 %d 个参数" % n)
        elif cmd in ("freeze", "unfreeze"):
            if not args:
                print("  用法: %s <正则>" % cmd)
            else:
                n = freeze(self.model, args[0], cmd == "freeze")
                print("  已处理 %d 个参数" % n)
        elif cmd == "nan":
            bad = [r.name for r in param_table(self.model) if r.nan > 0]
            print("  含 NaN 的参数: %s" % (bad if bad else "无"))
        else:
            print("  未知命令: %s（help 看帮助）" % cmd)
        return True

    def pause(self) -> None:
        """进入交互式循环。"""
        self.banner()
        while True:
            try:
                line = input("ail> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not self.execute(line):
                break
        print("[resume] 继续训练")

    def run_script(self, lines: Iterable[str]) -> None:
        """非交互：按脚本执行。

        脚本最后一步留空行表示退出，或以 resume 结尾。
        """
        self.banner()
        for line in lines:
            print("ail> %s" % line)
            if not self.execute(line):
                return


def pause(model: torch.nn.Module, *, step: Optional[int] = None,
          script: Optional[Iterable[str]] = None) -> None:
    """在训练代码两行之间插入这一句即可暂停。"""
    ins = Inspector(model, step=step)
    if script is not None:
        ins.run_script(script)
    else:
        ins.pause()
