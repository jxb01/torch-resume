"""torch_resume —— 模型训练干预工具箱。

只做一件事：让训练不用从头再来。

  Checkpoint         强制保存：最优 + 阶段性，原子写，异步，保留策略，磁盘限额
  diff / migrate     改结构之后继承权重与优化器状态
  detect_mismatch    换 loss 之后检测优化器状态失配（先试算再改）
  resume_with_edit   上面三件事的一条命令版本
  Inspector / pause  暂停、看权重、直接改

典型用法：

    import torch_resume as tl

    ck = tl.Checkpoint("runs/exp1", monitor="val_loss", every=500, best=3,
                       disk_limit_gb=20)
    for step, (xb, yb) in enumerate(loader):
        loss = train_step(xb, yb)
        ck.step(step, model, opt, metrics={"val_loss": loss.item()})

改完结构之后：

    tl.resume_with_edit(old_model, old_opt, new_model, new_opt,
                        loss_fn=my_loss, batches=sample_batches)
"""
from .checkpoint import Checkpoint
from .inspect import (
    Inspector,
    ParamStat,
    freeze,
    param_table,
    pause,
    print_params,
    resume_with_edit,
    set_param,
)
from .migrate import MigrateStats, apply_plan, migrate, migrate_optimizer
from .mismatch import MismatchReport, MismatchRow, detect_mismatch
from .plan import Action, MigrationPlan, diff
from .state import capture_rng, clone_tree, name_map, opt_state_by_name, restore_rng

__version__ = "0.1.0"

__all__ = [
    "Checkpoint",
    "Inspector", "pause", "param_table", "print_params",
    "set_param", "freeze", "resume_with_edit", "ParamStat",
    "diff", "MigrationPlan", "Action",
    "migrate", "migrate_optimizer", "apply_plan", "MigrateStats",
    "detect_mismatch", "MismatchReport", "MismatchRow",
    "capture_rng", "restore_rng", "clone_tree", "name_map", "opt_state_by_name",
]
