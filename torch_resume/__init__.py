"""torch_resume —— 模型训练干预工具箱。

只做一件事：让训练不用从头再来。

  Checkpoint          强制保存：最优 + 阶段性，原子写，异步，保留策略，磁盘限额
  diff / migrate      改结构之后继承权重与优化器状态
  detect_mismatch     换 loss 之后检测优化器状态失配（先试算再改）
  groups / PlanWarmup 按参数类别分组，并对全新/失配的参数自动预热
  surgery             改结构不用重写模型（widen / replace / insert / duplicate）
  Inspector / pause   暂停、看权重、直接改
  resume_with_edit    上面几件事的一条命令版本

典型用法：

    import torch_resume as tr

    ck = tr.Checkpoint("runs/exp1", monitor="val_loss", every=500, best=3,
                       disk_limit_gb=20)
    for step, (xb, yb) in enumerate(loader):
        loss = train_step(xb, yb)
        ck.step(step, model, opt, metrics={"val_loss": loss.item()})

把隐藏层加宽，然后接着训：

    new_model = tr.widen(model, "0", 128)          # 已经训练的部分原样保留
    new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    tr.resume_with_edit(model, opt, new_model, new_opt,
                        loss_fn=my_loss, batches=sample_batches,
                        warmup={"fresh": 500, "mismatched": 200})
"""
from ._io import auto_fix as _auto_fix, enable_utf8, safe_print
from .checkpoint import Checkpoint
from .groups import (
    KIND_FRESH,
    KIND_KEPT,
    KIND_MISMATCHED,
    KIND_PARTIAL,
    PlanWarmup,
    classify,
    group_report,
    retag_groups,
)
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
from .plan import SURGERY_ATTR, Action, MigrationPlan, diff, surgery_info
from .state import capture_rng, clone_tree, name_map, opt_state_by_name, restore_rng
from .surgery import (
    clone_model,
    describe as describe_surgery,
    duplicate_layer,
    freeze_module,
    insert_after,
    replace_module,
    widen,
)

__version__ = "0.2.1"

# Windows 控制台默认不是 UTF-8；只有在确实装不下中文时才自动切换，
# 正常的环境一行都不动。想禁用：TORCH_RESUME_NO_UTF8=1
_auto_fix()

__all__ = [
    # 检查点
    "Checkpoint",
    # 暂停 / 查看 / 修改
    "Inspector", "pause", "param_table", "print_params",
    "set_param", "freeze", "ParamStat", "resume_with_edit",
    # 结构对比与迁移
    "diff", "MigrationPlan", "Action", "surgery_info", "SURGERY_ATTR",
    "migrate", "migrate_optimizer", "apply_plan", "MigrateStats",
    # 失配检测
    "detect_mismatch", "MismatchReport", "MismatchRow",
    # 分组与预热
    "retag_groups", "PlanWarmup", "classify", "group_report",
    "KIND_KEPT", "KIND_PARTIAL", "KIND_FRESH", "KIND_MISMATCHED",
    # 模型手术
    "widen", "replace_module", "insert_after", "duplicate_layer",
    "freeze_module", "clone_model", "describe_surgery",
    # 状态工具
    "capture_rng", "restore_rng", "clone_tree", "name_map", "opt_state_by_name",
    # 输出编码
    "enable_utf8", "safe_print",
]
