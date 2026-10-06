"""模型结构缓存：改之前存一份，改坏了能变回去。

为什么不能只用 state_dict：**PyTorch 的 state_dict 只存权重，不存结构**。
要"变回去"得三样都在 —— 结构、权重、优化器状态（少最后一样续训会抖）。

三级恢复（按可靠性排序）：

  1. **结构一致** —— 就地 load_state_dict。最快最稳，推荐路径。
  2. **结构不一致** —— 用存档里 pickle 的模型对象重建（需要那个类能被反序列化）。
  3. **连对象都反序列化不了** —— **报错并给出差异**，不猜。

典型用法：

    cache = tr.ModelCache("runs/exp1/model_cache")

    # 改结构前先存一份
    cache.save(model, opt, tag="v1", note="加宽之前", metrics={"val_loss": 0.31})

    m2 = tr.widen(model, "0", 128)
    ...训练一会，发现更差了...

    model, opt = cache.restore("v1", model, opt)   # 变回去
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ._io import safe_print
from ._store import atomic_torch_save, atomic_write_json, read_json
from .plan import surgery_info
from .state import clone_tree


# ---------------------------------------------------------------- 结构描述

def describe_structure(model: nn.Module) -> Dict[str, Any]:
    """把结构压成可比较、可序列化的形式。"""
    shapes: Dict[str, List[int]] = {}
    n_params = 0
    for name, p in model.named_parameters():
        shapes[name] = list(p.shape)
        n_params += p.numel()
    for name, b in model.named_buffers():
        shapes.setdefault(name, list(b.shape))
    modules = [(name, type(m).__name__) for name, m in model.named_modules()]
    return {
        "param_shapes": shapes,
        "n_params": n_params,
        "modules": modules[:200],
        "n_modules": len(modules),
    }


def _shape_diff(expected: Dict[str, Any], actual: Dict[str, Any]) -> str:
    lines = []
    only_old = [k for k in expected if k not in actual]
    only_new = [k for k in actual if k not in expected]
    changed = [k for k in expected if k in actual and expected[k] != actual[k]]
    for k in only_old[:10]:
        lines.append("  存档里有、当前模型没有: %s %s" % (k, expected[k]))
    for k in only_new[:10]:
        lines.append("  当前模型有、存档没有: %s %s" % (k, actual[k]))
    for k in changed[:10]:
        lines.append("  形状不同: %s  %s -> %s" % (k, expected[k], actual[k]))
    for tag, arr in (("多出", only_old[10:]), ("多出", only_new[10:]), ("不同", changed[10:])):
        if arr:
            lines.append("  ... 另有 %d 项%s" % (len(arr), tag))
    return "\n".join(lines) if lines else "  （无差异）"


# ---------------------------------------------------------------- 存档元信息

@dataclass
class Snapshot:
    tag: str
    time: float
    note: str = ""
    step: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    n_params: int = 0
    n_modules: int = 0
    param_shapes: Dict[str, List[int]] = field(default_factory=dict)
    surgery: Dict[str, Any] = field(default_factory=dict)
    has_model_pickle: bool = False
    has_optimizer: bool = False

    @property
    def when(self) -> str:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(self.time))

    def metric_str(self) -> str:
        if not self.metrics:
            return "-"
        return " ".join("%s=%.5g" % (k, v) for k, v in sorted(self.metrics.items()))

    def line(self) -> str:
        return ("  %-18s %s  %8s 参数  step %-6s %s%s"
                % (self.tag, self.when, "%.2fM" % (self.n_params / 1e6)
                   if self.n_params >= 1e5 else str(self.n_params),
                   self.step if self.step is not None else "-",
                   self.metric_str(),
                   ("  # " + self.note) if self.note else ""))


# ---------------------------------------------------------------- 缓存

class ModelCache:
    """结构 + 权重 + 优化器状态的存档仓库。"""

    def __init__(
        self,
        dir: Union[str, Path],
        *,
        keep: int = 10,
        save_model_pickle: bool = True,
        verbose: bool = True,
    ) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.save_model_pickle = save_model_pickle
        self.verbose = verbose

    # ---------------- 路径 ----------------

    def _snap_dir(self, tag: str) -> Path:
        return self.dir / tag

    def _meta_path(self, tag: str) -> Path:
        return self._snap_dir(tag) / "meta.json"

    def _state_path(self, tag: str) -> Path:
        return self._snap_dir(tag) / "state.pt"

    def _model_path(self, tag: str) -> Path:
        return self._snap_dir(tag) / "model.pt"

    # ---------------- 写 ----------------

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        tag: Optional[str] = None,
        note: str = "",
        metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None,
    ) -> Snapshot:
        """存一份。tag 缺省用时间戳，重复的 tag 会被覆盖。"""
        tag = tag or time.strftime("snap-%Y%m%d-%H%M%S")
        d = self._snap_dir(tag)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

        desc = describe_structure(model)
        payload = {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": clone_tree(optimizer.state_dict()) if optimizer is not None else None,
            "rng": None,
        }
        atomic_torch_save(payload, self._state_path(tag))

        has_pickle = False
        if self.save_model_pickle:
            try:
                atomic_torch_save({"model": model}, self._model_path(tag))
                has_pickle = True
            except Exception as e:
                # 模型类不可序列化（例如定义在 __main__ 里的局部类）时不算失败，
                # 只是退化成"只能就地恢复"。
                if self.verbose:
                    safe_print("[model_cache] 模型对象无法序列化（%s），"
                               "只保留结构与权重" % type(e).__name__)

        snap = Snapshot(
            tag=tag, time=time.time(), note=note, step=step,
            metrics=dict(metrics or {}),
            n_params=desc["n_params"], n_modules=desc["n_modules"],
            param_shapes=desc["param_shapes"],
            surgery=dict(surgery_info(model)),
            has_model_pickle=has_pickle,
            has_optimizer=optimizer is not None,
        )
        atomic_write_json(asdict(snap), self._meta_path(tag))
        self._prune()
        if self.verbose:
            safe_print("[model_cache] 已存档 %s（%s）" % (tag, snap.metric_str()))
        return snap

    # ---------------- 读 ----------------

    def list(self) -> List[Snapshot]:
        out: List[Snapshot] = []
        if not self.dir.exists():
            return out
        for d in sorted(self.dir.iterdir()):
            if not d.is_dir():
                continue
            meta = read_json(self._meta_path(d.name))
            if not meta:
                continue
            try:
                out.append(Snapshot(**meta))
            except TypeError:
                continue
        out.sort(key=lambda s: s.time)
        return out

    def report(self) -> str:
        snaps = self.list()
        if not snaps:
            return "（模型缓存是空的）"
        lines = ["模型结构缓存  %s" % self.dir, "=" * 72]
        for s in snaps:
            lines.append(s.line())
        lines.append("-" * 72)
        lines.append("  共 %d 份，占用 %.1f MB"
                     % (len(snaps), self.size_bytes() / 1024 / 1024))
        return "\n".join(lines)

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.dir.rglob("*") if f.is_file())

    # ---------------- 恢复 ----------------

    def restore(
        self,
        tag: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        strict: bool = True,
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
        """回滚到某个存档。

        结构一致 -> 就地加载（推荐）；不一致 -> 用 pickle 的对象重建。
        返回 (model, optimizer, info)；info["mode"] 是 "inplace" 或 "rebuild"。
        """
        meta = self._read_meta(tag)
        if meta is None:
            raise KeyError("没有这个存档: %s（用 cache.report() 看有哪些）" % tag)
        state = torch.load(self._state_path(tag), map_location="cpu", weights_only=False)
        info: Dict[str, Any] = {"tag": tag, "mode": None, "note": meta.note,
                                "metrics": dict(meta.metrics), "step": meta.step}

        if model is None:
            rebuilt = self._load_pickled(tag)
            if rebuilt is None:
                raise FileNotFoundError(
                    "没给 model，且存档里没有可反序列化的模型对象（%s）" % tag)
            info["mode"] = "rebuild"
            if self.verbose:
                safe_print("[model_cache] 从存档对象重建模型（优化器需要你自己重建）")
            return rebuilt, None, info

        cur = describe_structure(model)
        if cur["param_shapes"] == meta.param_shapes:
            info["mode"] = "inplace"
        else:
            info["mode"] = "rebuild"
            diff = _shape_diff(meta.param_shapes, cur["param_shapes"])
            if strict:
                raise ValueError(
                    "目标模型结构与存档 %s 不一致，无法就地恢复：\n%s\n"
                    "（要么传入结构匹配的模型，要么用 strict=False 走重建路径）"
                    % (tag, diff))
            rebuilt = self._load_pickled(tag)
            if rebuilt is None:
                raise FileNotFoundError(
                    "结构不一致，且存档里没有可反序列化的模型对象：\n%s" % diff)
            if self.verbose:
                safe_print("[model_cache] 结构不一致，已从存档对象重建：\n" + diff)
            return rebuilt, None, info

        model.load_state_dict(state["model"], strict=True)
        if optimizer is not None and state.get("optimizer"):
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception as e:
                info["optimizer_error"] = str(e)
                if self.verbose:
                    safe_print("[model_cache] 优化器状态加载失败（权重已恢复）：%s" % e)
        if self.verbose:
            safe_print("[model_cache] 已回滚到 %s（就地，结构一致）" % tag)
        return model, optimizer, info

    def load_weights_only(self, tag: str) -> Dict[str, torch.Tensor]:
        """只要权重（用来手工塞进别的模型）。"""
        state = torch.load(self._state_path(tag), map_location="cpu", weights_only=False)
        return state["model"]

    def diff(self, tag_a: str, tag_b: str) -> str:
        """比较两个存档的结构。"""
        a, b = self._read_meta(tag_a), self._read_meta(tag_b)
        if a is None or b is None:
            raise KeyError("存档不存在: %s" % ("%s / %s" % (tag_a, tag_b)))
        body = _shape_diff(a.param_shapes, b.param_shapes)
        lines = ["结构对比  %s -> %s" % (tag_a, tag_b), "=" * 72,
                 "  参数量: %s -> %s" % (a.n_params, b.n_params),
                 "  模块数: %s -> %s" % (a.n_modules, b.n_modules)]
        if a.surgery or b.surgery:
            lines.append("  %s 的手术记录: %s" % (tag_a, a.surgery.get("notes") or "无"))
            lines.append("  %s 的手术记录: %s" % (tag_b, b.surgery.get("notes") or "无"))
        lines.append(body)
        return "\n".join(lines)

    # ---------------- 维护 ----------------

    def delete(self, tag: str) -> None:
        d = self._snap_dir(tag)
        if d.exists():
            shutil.rmtree(d)

    def _prune(self) -> None:
        if self.keep <= 0:
            return
        snaps = self.list()
        for s in snaps[: max(0, len(snaps) - self.keep)]:
            self.delete(s.tag)

    # ---------------- 内部 ----------------

    def _read_meta(self, tag: str) -> Optional[Snapshot]:
        meta = read_json(self._meta_path(tag))
        if not meta:
            return None
        try:
            return Snapshot(**meta)
        except TypeError:
            return None

    def _load_pickled(self, tag: str) -> Optional[nn.Module]:
        p = self._model_path(tag)
        if not p.exists():
            return None
        try:
            return torch.load(p, map_location="cpu", weights_only=False)["model"]
        except Exception as e:
            if self.verbose:
                safe_print("[model_cache] 反序列化失败（模型类可能改了或不在路径上）: %s" % e)
            return None
