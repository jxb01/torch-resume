"""强制检查点：最优 + 阶段性，原子写，异步保存，保留策略，磁盘限额。

设计要点（都是踩过坑才有的）：
  1. state_dict() 返回的是**引用**，异步保存前必须 clone，否则存下来的是"当时的指针、之后的值"
  2. 必须**原子写**：先写临时文件再 os.replace，否则写一半崩了会毁掉检查点
  3. 快照必须含**随机数状态**，否则续训不可复现
  4. 最优模型存**完整训练状态**（便于续训），同时导出**纯权重**（便于部署/分享）
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from .state import capture_rng, clone_tree, restore_rng

CKPT_EXT = ".ailck"
WEIGHT_EXT = ".ailw"


class Checkpoint:
    """检查点管理器。用法：

        ck = Checkpoint("runs/exp1", monitor="val_loss", every=500, best=3)
        for step, ... in enumerate(loader):
            ...
            ck.step(step, model, optimizer, metrics={"val_loss": v})
    """

    def __init__(
        self,
        dir: Union[str, Path],
        *,
        monitor: Optional[str] = None,
        mode: str = "min",
        best: int = 3,
        every: Optional[int] = None,
        keep_last: int = 2,
        disk_limit_gb: Optional[float] = None,
        async_save: bool = True,
        export_weights: bool = True,
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode 必须是 min 或 max")
        if every is None and best <= 0:
            raise ValueError("every 和 best 不能同时缺省 —— 否则没有保存策略")
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best = best
        self.every = every
        self.keep_last = keep_last
        self.disk_limit_gb = disk_limit_gb
        self.async_save = async_save
        self.export_weights = export_weights
        self.verbose = verbose

        self._best: List[Dict[str, Any]] = []      # [{step, metric, path}]
        self._periodic: List[Dict[str, Any]] = []
        self._q: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._load_index()

    # ---------------- 内部 ----------------

    def _index_path(self) -> Path:
        return self.dir / "index.json"

    def _load_index(self) -> None:
        p = self._index_path()
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                self._best = d.get("best", [])
                self._periodic = d.get("periodic", [])
            except Exception:
                pass

    def _save_index(self) -> None:
        d = {"best": self._best, "periodic": self._periodic}
        tmp = self._index_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self._index_path())

    def _used_gb(self) -> float:
        total = sum(f.stat().st_size for f in self.dir.glob("*") if f.is_file())
        return total / (1024 ** 3)

    def _check_disk(self, size_hint: float = 0.0) -> bool:
        if self.disk_limit_gb is None:
            return True
        used = self._used_gb() + size_hint
        if used > self.disk_limit_gb:
            if self.verbose:
                print("[checkpoint] 磁盘限额 %.2f GB 已用 %.2f GB，先清理"
                      % (self.disk_limit_gb, used))
            self._prune(force=True)
            used = self._used_gb() + size_hint
            if used > self.disk_limit_gb:
                print("[checkpoint] 警告：仍超出限额，本次跳过保存")
                return False
        return True

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        def run() -> None:
            while True:
                item = self._q.get()
                if item is None:
                    break
                path, payload = item
                try:
                    _atomic_torch_save(payload, path)
                except Exception as e:      # 保存失败不能让训练崩
                    print("[checkpoint] 保存失败 %s: %s" % (path.name, e))

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()

    def _submit(self, payload: Dict[str, Any], path: Path) -> None:
        if self.async_save:
            self._start_worker()
            self._q.put((path, payload))
        else:
            _atomic_torch_save(payload, path)

    # ---------------- 快照 ----------------

    def snapshot(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        scheduler: Any = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """抓一份**独立**的训练状态。必须在训练线程上调用（clone 在这里发生）。"""
        return {
            "step": step,
            "model": {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()},
            "optimizer": clone_tree(optimizer.state_dict()) if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": capture_rng(),
            "metrics": dict(metrics or {}),
            "extra": dict(extra or {}),
            "saved_at": time.time(),
            "format": 1,
        }

    # ---------------- 主入口 ----------------

    def step(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        scheduler: Any = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """按策略决定是否保存。返回已提交的路径（异步时表示已排队）。"""
        metrics = metrics or {}
        want_periodic = self.every is not None and step % self.every == 0
        metric = metrics.get(self.monitor) if self.monitor else None
        want_best = self.monitor is not None and metric is not None and self.best > 0
        if not (force or want_periodic or want_best):
            return None
        if not self._check_disk():
            return None

        payload = self.snapshot(step, model, optimizer, scheduler=scheduler,
                                metrics=metrics, extra=extra)

        saved: Optional[Path] = None
        if want_periodic or force:
            p = self.dir / ("step-%08d%s" % (step, CKPT_EXT))
            self._submit(payload, p)
            with self._lock:
                self._periodic.append({"step": step, "path": p.name})
                self._save_index()
            saved = p
            if self.verbose:
                print("[checkpoint] 阶段性保存 step=%d -> %s" % (step, p.name))

        if want_best:
            is_better = (len(self._best) < self.best) or self._better(
                metric, [b["metric"] for b in self._best])
            if is_better:
                p = self.dir / ("best-step%08d%s" % (step, CKPT_EXT))
                self._submit(payload, p)
                with self._lock:
                    self._best.append({"step": step, "metric": float(metric),
                                       "path": p.name})
                    self._best.sort(key=lambda b: b["metric"],
                                    reverse=(self.mode == "max"))
                    self._best = self._best[: self.best]
                    self._save_index()
                saved = saved or p
                if self.verbose:
                    print("[checkpoint] 最优更新 %s=%.6f step=%d -> %s"
                          % (self.monitor, metric, step, p.name))
                if self.export_weights:
                    self._export_weights(payload, p.with_suffix(WEIGHT_EXT))

        self._prune()
        return saved

    def _better(self, metric: float, existing: List[float]) -> bool:
        if not existing:
            return True
        return metric > min(existing) if self.mode == "max" else metric < max(existing)

    def _export_weights(self, payload: Dict[str, Any], path: Path) -> None:
        try:
            _atomic_torch_save({"model": payload["model"], "step": payload["step"]}, path)
        except Exception as e:
            print("[checkpoint] 导出纯权重失败: %s" % e)

    # ---------------- 保留策略 ----------------

    def _prune(self, force: bool = False) -> None:
        with self._lock:
            keep_p = self._periodic[-self.keep_last:] if self.keep_last else []
            drop_p = [x for x in self._periodic if x not in keep_p]
            self._periodic = keep_p
            keep_b = {b["path"] for b in self._best}
            self._save_index()
        for x in drop_p:
            _unlink(self.dir / x["path"])
        for f in list(self.dir.glob("*" + CKPT_EXT)):
            name = f.name
            in_best = name in keep_b
            in_periodic = any(p["path"] == name for p in self._periodic)
            if not (in_best or in_periodic):
                _unlink(f)
        for f in list(self.dir.glob("*" + WEIGHT_EXT)):
            base = f.name.replace(WEIGHT_EXT, CKPT_EXT)
            if base not in keep_b:
                _unlink(f)

    # ---------------- 恢复 ----------------

    def best_path(self) -> Optional[Path]:
        if not self._best:
            return None
        return self.dir / self._best[0]["path"]

    def latest_path(self) -> Optional[Path]:
        cands = list(self.dir.glob("*" + CKPT_EXT))
        if not cands:
            return None
        return max(cands, key=lambda f: f.stat().st_mtime)

    def load(
        self,
        path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        scheduler: Any = None,
        restore_random: bool = True,
    ) -> Dict[str, Any]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if restore_random and payload.get("rng"):
            restore_rng(payload["rng"])
        return payload

    def resume_hint(self) -> Optional[Dict[str, Any]]:
        """启动时调用：有检查点就问要不要恢复。"""
        p = self.latest_path()
        if p is None:
            return None
        info = {"path": str(p)}
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
            info["step"] = payload.get("step")
            info["metrics"] = payload.get("metrics", {})
        except Exception:
            pass
        return info


def _unlink(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    """先写临时文件再 rename —— 写一半崩了也不会毁掉已有检查点。"""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
