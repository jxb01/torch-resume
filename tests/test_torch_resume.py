# -*- coding: utf-8 -*-
"""torch_resume 测试。不依赖 pytest，直接跑：

    python tests/test_torch_resume.py

强制 CPU（CI 上没有 GPU）：

    TORCH_RESUME_CPU=1 python tests/test_torch_resume.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

import torch_resume as tl

DEV = "cuda" if (torch.cuda.is_available()
                    and os.environ.get("TORCH_RESUME_CPU") != "1") else "cpu"

RESULTS = []


def case(fn):
    try:
        fn()
        RESULTS.append((fn.__name__, "PASS", ""))
        print("  PASS  %s" % fn.__name__)
    except Exception as e:
        RESULTS.append((fn.__name__, "FAIL", str(e)))
        print("  FAIL  %s -> %s" % (fn.__name__, e))
        traceback.print_exc()
    return fn


def mlp(dims, seed=0):
    torch.manual_seed(seed)
    layers, prev = [], dims[0]
    for d in dims[1:]:
        layers += [nn.Linear(prev, d), nn.ReLU()]
        prev = d
    return nn.Sequential(*layers[:-1]).to(DEV)


def train_steps(model, opt, n=5, batch=32, seed=1, scale=1.0):
    torch.manual_seed(seed)
    x = torch.randn(batch, model[0].in_features, device=DEV)
    y = torch.randn(batch, model[-1].out_features, device=DEV)
    for _ in range(n):
        loss = ((model(x) - y) ** 2).mean() * scale
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item())


# ------------------------------------------------------------------ 结构 diff

@case
def test_plan_categories():
    a = mlp([8, 16, 4])
    b = mlp([8, 16, 4, 2])          # 加了一层
    old = {k: v.detach().cpu() for k, v in a.state_dict().items()}
    plan = tl.diff(b, old)
    kinds = plan.summary
    assert kinds.get("keep", 0) == 4, "fc1 与 fc2 的 w/b 都应保留, got %s" % kinds
    assert kinds.get("reset", 0) == 2, "新层的 w/b 应重置, got %s" % kinds
    assert "drop" not in kinds


@case
def test_plan_resize_partial():
    a = mlp([8, 4])
    b = mlp([8, 6])
    old = {k: v.detach().cpu() for k, v in a.state_dict().items()}
    plan = tl.diff(b, old)
    assert plan.summary.get("partial", 0) == 2, plan.summary
    assert plan.warnings, "partial 必须产出警告"


@case
def test_partial_min_ratio_guard():
    """重叠比例太低时必须宁可从零初始化。

    Linear(8,4) -> Linear(8,32)：每维都不小于，但重叠只有 12.5%。
    把旧的输出层搬进新的隐藏层是语义错误。
    """
    a = mlp([8, 4])
    b = mlp([8, 32])
    old = {k: v.detach().cpu() for k, v in a.state_dict().items()}
    plan = tl.diff(b, old)
    assert plan.summary.get("reset", 0) == 2, plan.summary
    assert not plan.summary.get("partial"), "重叠不足时不该切片继承: %s" % plan.summary
    plan2 = tl.diff(b, old, partial_min_ratio=0.0)
    assert plan2.summary.get("partial", 0) == 2, plan2.summary


@case
def test_partial_min_ratio_allows_widening():
    """加宽一倍（重叠 50%）应当允许继承 —— 这是最有价值的用法。"""
    a = mlp([8, 64, 4])
    b = mlp([8, 128, 4])
    old = {k: v.detach().cpu() for k, v in a.state_dict().items()}
    plan = tl.diff(b, old)
    assert plan.summary.get("partial", 0) >= 2, plan.summary


@case
def test_plan_incompatible_reset():
    a = mlp([8, 4])
    b = mlp([8, 3])
    old = {k: v.detach().cpu() for k, v in a.state_dict().items()}
    plan = tl.diff(b, old)
    assert plan.summary.get("reset", 0) == 2, plan.summary


# ------------------------------------------------------------------ 权重继承

@case
def test_weight_inheritance_exact():
    a = mlp([8, 16, 4])
    opt = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, opt, n=8)
    b = mlp([8, 16, 4, 2])          # 新层用不同 seed 初始化
    before = {k: v.detach().clone() for k, v in b.state_dict().items()}
    old_opt = torch.optim.Adam(b.parameters(), lr=1e-3)

    st = tl.migrate(b, old_opt, a, opt, verbose=False)
    assert st.kept == 4 and st.reset == 2, st.report()

    for k in a.state_dict():
        assert torch.equal(b.state_dict()[k], a.state_dict()[k]), \
            "保留的参数必须逐位相同: %s" % k
    # 新层不应等于迁移前的值（不应被意外改动）
    new_keys = [k for k in b.state_dict() if k not in a.state_dict()]
    for k in new_keys:
        assert torch.equal(b.state_dict()[k], before[k]), "新参数不该被动过: %s" % k


@case
def test_partial_slice_inheritance():
    a = mlp([8, 4])
    opt = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, opt, n=4)
    b = mlp([8, 6])
    old_opt = torch.optim.Adam(b.parameters(), lr=1e-3)
    tl.migrate(b, old_opt, a, opt, verbose=False)

    w_old = a[0].weight.detach()
    w_new = b[0].weight.detach()
    assert w_new.shape == (6, 8)
    assert torch.equal(w_new[:4], w_old), "前 4 行应按切片继承"
    b_old, b_new = a[0].bias.detach(), b[0].bias.detach()
    assert torch.equal(b_new[:4], b_old)


# ------------------------------------------------------------------ 优化器状态

@case
def test_optimizer_state_migrated():
    a = mlp([8, 16, 4])
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=6)
    b = mlp([8, 16, 4])
    ob = torch.optim.Adam(b.parameters(), lr=1e-3)
    tl.migrate(b, ob, a, oa, verbose=False)

    n_checked = 0
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        sa, sb = oa.state.get(pa), ob.state.get(pb)
        assert sa is not None and sb is not None, nb
        assert "exp_avg" in sb and "exp_avg_sq" in sb, "动量应被迁移: %s" % nb
        assert torch.allclose(sa["exp_avg"], sb["exp_avg"]), nb
        assert torch.allclose(sa["exp_avg_sq"], sb["exp_avg_sq"]), nb
        n_checked += 1
    assert n_checked == 4


@case
def test_optimizer_state_partial_slice():
    a = mlp([8, 4])
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=6)
    b = mlp([8, 6])
    ob = torch.optim.Adam(b.parameters(), lr=1e-3)
    tl.migrate(b, ob, a, oa, verbose=False)

    pa, pb = a[0].weight, b[0].weight
    va = oa.state[pa]["exp_avg_sq"]
    vb = ob.state[pb]["exp_avg_sq"]
    assert vb.shape == (6, 8)
    assert torch.allclose(vb[:4], va), "动量应按同样切片继承"


@case
def test_partial_optimizer_state_nonnegative():
    """回归测试：切片继承 exp_avg_sq 时，切片以外必须是 0。

    曾经的 bug：那里被填进了**参数值**，而参数有负值 -> sqrt -> NaN，
    并且失配检测把 NaN 当成 ok 静默放过，要续训到第 2 步才炸。
    """
    a = mlp([8, 4])
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=6)
    b = mlp([8, 6])
    ob = torch.optim.Adam(b.parameters(), lr=1e-3)
    tl.migrate(b, ob, a, oa, verbose=False)

    checked = 0
    for name, p in b.named_parameters():
        st = ob.state.get(p)
        if st and "exp_avg_sq" in st:
            v = st["exp_avg_sq"]
            checked += 1
            assert torch.isfinite(v).all(), "%s 的 exp_avg_sq 含非有限值" % name
            assert (v >= 0).all(), "%s 的 exp_avg_sq 含负值" % name
    assert checked >= 2, "应至少迁移了两个参数的优化器状态"


@case
def test_mismatch_flags_nonfinite():
    """非有限值必须被标成 invalid，不能被当成 ok。"""
    a = mlp([8, 4])
    opt = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, opt, n=4)
    # 人为注入 NaN
    with torch.no_grad():
        for p in a.parameters():
            if p in opt.state:
                opt.state[p]["exp_avg_sq"].fill_(float("nan"))
    x = torch.randn(16, 8, device=DEV)
    y = torch.randn(16, 4, device=DEV)
    rep = tl.detect_mismatch(a, opt, lambda m, b: ((m(b[0]) - b[1]) ** 2).mean(),
                             [(x, y)] * 2)
    assert rep.n_invalid == len(rep.rows), "全部应被判为 invalid: %s" % rep.report()
    n = rep.apply(opt, policy="threshold")
    assert n > 0, "invalid 的参数应被重置"


# ------------------------------------------------------------------ 失配检测

@case
def test_mismatch_detects_stall():
    """loss 缩小 100 倍 -> 新梯度变小 -> v 偏大 -> 会停滞"""
    a = mlp([8, 16, 4])
    opt = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, opt, n=10, scale=1.0)

    torch.manual_seed(7)
    x = torch.randn(32, 8, device=DEV)
    y = torch.randn(32, 4, device=DEV)

    def small_loss(m, b):
        xb, yb = b
        return ((m(xb) - yb) ** 2).mean() * 0.01

    rep = tl.detect_mismatch(a, opt, small_loss, [(x, y)] * 4, threshold=4.0)
    assert rep.n_stall > 0, "应检测到停滞风险: %s" % rep.report()
    print("      " + rep.report().replace("\n", "\n      "))


@case
def test_mismatch_detects_oscillate():
    """loss 放大 100 倍 -> 新梯度变大 -> v 偏小 -> 会震荡"""
    a = mlp([8, 16, 4])
    opt = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, opt, n=10, scale=1.0)

    torch.manual_seed(7)
    x = torch.randn(32, 8, device=DEV)
    y = torch.randn(32, 4, device=DEV)

    def big_loss(m, b):
        xb, yb = b
        return ((m(xb) - yb) ** 2).mean() * 100.0

    rep = tl.detect_mismatch(a, opt, big_loss, [(x, y)] * 4, threshold=4.0)
    assert rep.n_oscillate > 0, "应检测到震荡风险: %s" % rep.report()


@case
def test_mismatch_apply():
    a = mlp([8, 16, 4])
    opt = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, opt, n=10)
    torch.manual_seed(7)
    x = torch.randn(32, 8, device=DEV)
    y = torch.randn(32, 4, device=DEV)

    def scaled(m, b):
        xb, yb = b
        return ((m(xb) - yb) ** 2).mean() * 0.01

    rep = tl.detect_mismatch(a, opt, scaled, [(x, y)] * 4)
    before = {n: opt.state[p]["exp_avg_sq"].clone()
              for n, p in a.named_parameters()}
    n = rep.apply(opt, policy="threshold")
    assert n > 0, "应重置至少一个参数的 exp_avg_sq"
    changed = sum(1 for nm, p in a.named_parameters()
                  if not torch.equal(opt.state[p]["exp_avg_sq"], before[nm]))
    assert changed == n


@case
def test_mismatch_apply_all():
    a = mlp([8, 16, 4])
    opt = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, opt, n=6)
    x = torch.randn(16, 8, device=DEV)
    y = torch.randn(16, 4, device=DEV)
    rep = tl.detect_mismatch(a, opt, lambda m, b: ((m(b[0]) - b[1]) ** 2).mean(),
                             [(x, y)] * 2)
    n = rep.apply(opt, policy="all")
    assert n == 4, "policy=all 应处理全部 4 个参数, got %d" % n


# ------------------------------------------------------------------ 检查点

@case
def test_checkpoint_roundtrip():
    d = tempfile.mkdtemp(prefix="tl_ck_")
    try:
        a = mlp([8, 16, 4])
        opt = torch.optim.Adam(a.parameters(), lr=1e-3)
        ck = tl.Checkpoint(d, monitor="loss", every=2, best=2,
                           keep_last=2, async_save=False, verbose=False)
        for step in range(1, 7):
            train_steps(a, opt, n=1)
            ck.step(step, a, opt, metrics={"loss": 1.0 / step})

        snap = {k: v.clone() for k, v in a.state_dict().items()}
        p = ck.latest_path()
        assert p is not None, "应有检查点"

        b = mlp([8, 16, 4], seed=99)
        ob = torch.optim.Adam(b.parameters(), lr=1e-3)
        payload = ck.load(p, b, ob)
        for k in snap:
            assert torch.equal(b.state_dict()[k], snap[k]), k
        assert payload["step"] == 6
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case
def test_checkpoint_best_tracking():
    d = tempfile.mkdtemp(prefix="tl_ck_")
    try:
        a = mlp([8, 4])
        opt = torch.optim.Adam(a.parameters(), lr=1e-3)
        ck = tl.Checkpoint(d, monitor="loss", mode="min", every=100, best=2,
                           async_save=False, verbose=False)
        for step, v in enumerate([5.0, 3.0, 1.0, 4.0, 2.0], start=1):
            ck.step(step, a, opt, metrics={"loss": v})
        best = ck._best
        assert len(best) == 2, best
        assert best[0]["metric"] == 1.0, "最优应是 1.0: %s" % best
        assert best[1]["metric"] == 2.0, "次优应是 2.0: %s" % best
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case
def test_checkpoint_retention_and_atomic():
    d = tempfile.mkdtemp(prefix="tl_ck_")
    try:
        a = mlp([8, 4])
        opt = torch.optim.Adam(a.parameters(), lr=1e-3)
        ck = tl.Checkpoint(d, monitor="loss", every=1, best=2, keep_last=2,
                           async_save=False, verbose=False)
        for step in range(1, 9):
            ck.step(step, a, opt, metrics={"loss": float(step % 3)})
        files = sorted(f.name for f in ck.dir.glob("*" + ".ailck"))
        assert not any(f.endswith(".tmp") for f in os.listdir(ck.dir)), "不该留下 .tmp"
        periodic = [f for f in files if f.startswith("step-")]
        assert len(periodic) <= 2, "阶段性最多留 2 份: %s" % periodic
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case
def test_checkpoint_rng_restore():
    d = tempfile.mkdtemp(prefix="tl_ck_")
    try:
        a = mlp([8, 4])
        ck = tl.Checkpoint(d, every=1, best=0, keep_last=1,
                           async_save=False, verbose=False)
        torch.manual_seed(1234)
        ck.step(1, a, None)              # 记下此刻的随机数状态
        expect = torch.randn(4)          # 本应抽到的数
        torch.manual_seed(999)           # 把随机数状态打乱
        _ = torch.randn(16)
        ck.load(ck.latest_path(), a, None, restore_random=True)
        got = torch.randn(4)
        assert torch.allclose(expect, got), "随机数状态应被恢复"
    finally:
        shutil.rmtree(d, ignore_errors=True)


@case
def test_inspector_script():
    a = mlp([8, 16, 4])
    tl.Inspector(a, step=5).run_script(
        ["list 0", "zero 0.bias", "freeze 0.weight", "nan", "resume"])
    assert float(a[0].bias.abs().sum()) == 0.0, "zero 命令应生效"
    assert a[0].weight.requires_grad is False, "freeze 命令应生效"


@case
def test_checkpoint_async():
    d = tempfile.mkdtemp(prefix="tl_ck_")
    try:
        a = mlp([8, 4])
        ck = tl.Checkpoint(d, every=1, best=0, keep_last=1,
                           async_save=True, verbose=False)
        ck.step(1, a)
        import time
        for _ in range(50):
            if ck.latest_path() is not None:
                break
            time.sleep(0.05)
        assert ck.latest_path() is not None, "异步保存应最终落盘"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ 端到端

@case
def test_resume_with_edit_end_to_end():
    """训练 -> 改结构 -> 继承 -> 续训，验证不崩且 loss 有限。"""
    a = mlp([8, 16, 4])
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=20)

    b = mlp([8, 16, 4, 4])          # 加了一层
    ob = torch.optim.Adam(b.parameters(), lr=1e-3)

    torch.manual_seed(3)
    x = torch.randn(32, 8, device=DEV)
    y = torch.randn(32, 4, device=DEV)

    def loss_fn(m, batch):
        xb, yb = batch
        return ((m(xb) - yb) ** 2).mean()

    out = tl.resume_with_edit(a, oa, b, ob, loss_fn=loss_fn,
                              batches=[(x, y)] * 4, verbose=False)
    assert out["migrate"].kept == 4
    assert out["migrate"].opt_migrated == 4, out["migrate"].report()

    # 续训几步，不应出现 NaN/Inf
    for _ in range(10):
        loss = loss_fn(b, (x, y))
        ob.zero_grad()
        loss.backward()
        ob.step()
    assert torch.isfinite(loss), "续训后 loss 应有限，实际 %s" % loss.item()

    # 保留的参数在续训后仍应远离随机初始化（说明确实继承了）
    assert torch.isfinite(b[0].weight).all()


@case
def test_resume_with_edit_after_loss_change():
    """整个流程：训练 -> 换 loss -> 检测失配 -> 处理 -> 续训。"""
    a = mlp([8, 16, 4])
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=20, scale=1.0)
    b = mlp([8, 16, 4])
    ob = torch.optim.Adam(b.parameters(), lr=1e-3)

    torch.manual_seed(5)
    x = torch.randn(32, 8, device=DEV)
    y = torch.randn(32, 4, device=DEV)

    # 换了 loss 的尺度（模拟换了 loss 类型导致的尺度剧变）
    def new_loss(m, batch):
        xb, yb = batch
        return ((m(xb) - yb) ** 2).mean() * 0.02

    out = tl.resume_with_edit(a, oa, b, ob, loss_fn=new_loss,
                              batches=[(x, y)] * 4, threshold=4.0,
                              policy="threshold", verbose=False)
    rep = out["mismatch"]
    assert rep is not None
    assert rep.n_stall + rep.n_oscillate > 0, "应检测到失配: %s" % rep.report()


# ------------------------------------------------------------------ 模型手术

@case
def test_surgery_widen_keeps_trained():
    """加宽隐藏层：已训练的行必须原样保留，旧的模型不能被动过。"""
    a = mlp([8, 64, 4])
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=8)

    b = tl.widen(a, "0", 128)

    assert a[0].weight.shape == (64, 8), "widen 不该改动原模型"
    assert b[0].weight.shape == (128, 8)
    assert torch.equal(b[0].weight[:64], a[0].weight.detach())
    assert b[2].weight.shape == (4, 128)
    assert torch.equal(b[2].weight[:, :64], a[2].weight.detach())

    # 迁移时应当被认定为显式继承，不受重叠比例门槛限制
    ob = torch.optim.Adam(b.parameters(), lr=1e-3)
    st = tl.migrate(b, ob, a, oa, verbose=False)
    assert st.partial >= 2, "加宽后的层应做切片继承: %s" % st.report()
    assert st.reset == 0, "不该有参数被重置: %s" % st.report()


@case
def test_surgery_widen_resizes_norm():
    """中间夹了 LayerNorm 也要跟着变宽。"""
    a = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 4))
    b = tl.widen(a, "0", 128)
    assert b[0].weight.shape == (128, 8)
    assert tuple(b[2].normalized_shape) == (128,)
    assert b[3].weight.shape == (4, 128)


@case
def test_surgery_widen_rejects_bad_input():
    a = mlp([8, 16, 4])
    try:
        tl.widen(a, "0", 8)          # 没有变大
        raise AssertionError("应该报错")
    except ValueError:
        pass
    try:
        tl.widen(a, "1", 32)         # ReLU 不是 Linear
        raise AssertionError("应该报错")
    except TypeError:
        pass


@case
def test_surgery_replace_module():
    a = mlp([8, 16, 4])
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=4)
    b = tl.replace_module(a, "0", nn.Linear(8, 16))
    assert torch.equal(b[0].weight.detach(), a[0].weight.detach()), \
        "形状一致时迁移应逐位保留"
    plan = tl.diff(b, a.state_dict())
    assert plan.summary.get("keep", 0) == 4, plan.summary


@case
def test_surgery_insert_after_renames():
    """在 Sequential 中间插入一层，后面的下标位移要能被自动对齐。"""
    a = mlp([8, 16, 4])          # [Linear(8,16), ReLU, Linear(16,4)]
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=4)

    b = tl.insert_after(a, "1", nn.ReLU())   # 变成 [L, ReLU, ReLU, L]
    assert len(b) == 4 and isinstance(b[2], nn.ReLU)

    plan = tl.diff(b, a.state_dict())
    assert plan.summary.get("keep", 0) == 4, "位移应被正确对齐: %s" % plan.summary
    assert not plan.summary.get("reset"), "不该有参数被重置: %s" % plan.summary


@case
def test_surgery_duplicate_layer_copies_weights():
    """扩深的份必须复制原层权重，而不是随机初始化。"""
    torch.manual_seed(0)
    a = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).to(DEV)
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=6)

    b = tl.duplicate_layer(a, "0", 1)     # [L, L, ReLU, L]
    assert len(b) == 4
    plan = tl.diff(b, a.state_dict())
    assert not plan.summary.get("reset"), "复制来的份不该被重置: %s" % plan.summary

    # 迁移后，第 1 层应当与第 0 层逐位相同
    ob = torch.optim.Adam(b.parameters(), lr=1e-3)
    tl.migrate(b, ob, a, oa, verbose=False)
    assert torch.equal(b[1].weight.detach(), b[0].weight.detach())


@case
def test_surgery_describe():
    a = mlp([8, 64, 4])
    b = tl.widen(a, "0", 128)
    txt = tl.describe_surgery(b)
    assert "widen" in txt, txt


# ------------------------------------------------------------------ 分组与预热

@case
def test_groups_classify():
    plan = tl.MigrationPlan(actions=[
        tl.Action("0.weight", "keep", ""), tl.Action("0.bias", "keep", ""),
        tl.Action("2.weight", "partial", ""), tl.Action("2.bias", "reset", ""),
    ])
    m = mlp([8, 16, 4])
    kind = tl.classify(m, plan)
    assert kind["0.weight"] == tl.KIND_KEPT
    assert kind["2.weight"] == tl.KIND_PARTIAL
    assert kind["2.bias"] == tl.KIND_FRESH


@case
def test_groups_retag_preserves_optimizer_state():
    """重新分组不能丢掉已迁移的动量。"""
    a = mlp([8, 16, 4])
    oa = torch.optim.Adam(a.parameters(), lr=1e-3)
    train_steps(a, oa, n=6)
    b = mlp([8, 16, 4, 2])
    ob = torch.optim.Adam(b.parameters(), lr=1e-3)
    st = tl.migrate(b, ob, a, oa, verbose=False)

    before = {id(p): ob.state[p]["exp_avg"].clone()
              for p in b.parameters() if p in ob.state}
    groups = tl.retag_groups(ob, b, st.plan, lr_by_kind={"fresh": 2.0})
    assert len(groups) >= 2, "应当分出 fresh 和 kept 两组"
    for p in b.parameters():
        if id(p) in before:
            assert torch.equal(ob.state[p]["exp_avg"], before[id(p)]), \
                "重新分组不该动 optimizer.state"


@case
def test_warmup_ramps_lr():
    m = mlp([8, 16, 4])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    plan = tl.MigrationPlan(actions=[
        tl.Action("0.weight", "reset", ""), tl.Action("0.bias", "reset", ""),
        tl.Action("2.weight", "keep", ""), tl.Action("2.bias", "keep", ""),
    ])
    tl.retag_groups(opt, m, plan, lr_by_kind={"fresh": 2.0})
    fresh = [g for g in opt.param_groups if g["plan_kind"] == tl.KIND_FRESH][0]
    assert abs(fresh["lr"] - 2e-3) < 1e-12, fresh["lr"]

    sched = tl.PlanWarmup(opt, {"fresh": 10}, start_factor=0.1)
    assert abs(fresh["lr"] - 2e-4) < 1e-12, "起点应是基准的 0.1 倍: %s" % fresh["lr"]
    for _ in range(10):
        sched.step()
    assert abs(fresh["lr"] - 2e-3) < 1e-9, "10 步后应回到基准: %s" % fresh["lr"]
    assert "fresh" in sched.report()


@case
def test_resume_with_edit_warmup_end_to_end():
    """完整流程：加宽 -> 迁移 -> 分组 -> 预热 -> 续训不炸。"""
    a = mlp([8, 64, 4])
    oa = torch.optim.Adam(a.parameters(), lr=3e-3)
    train_steps(a, oa, n=30)

    b = tl.widen(a, "0", 128)
    ob = torch.optim.Adam(b.parameters(), lr=3e-3)
    x = torch.randn(32, 8, device=DEV)
    y = torch.randn(32, 4, device=DEV)

    def loss_fn(mm, batch):
        return ((mm(batch[0]) - batch[1]) ** 2).mean()

    out = tl.resume_with_edit(a, oa, b, ob, loss_fn=loss_fn,
                              batches=[(x, y)] * 4,
                              warmup={"fresh": 20, "mismatched": 10},
                              lr_by_kind={"fresh": 2.0}, verbose=False)
    assert out["scheduler"] is not None, "应返回 warmup 调度器"
    kinds = {g.get("plan_kind") for g in ob.param_groups}
    # 加宽不会产生全新参数（全部是 partial / keep），所以断言"有需要特殊处理的组"
    assert kinds - {tl.KIND_KEPT}, "应至少有一个非 kept 组: %s" % kinds

    for _ in range(25):
        loss = loss_fn(b, (x, y))
        ob.zero_grad()
        loss.backward()
        ob.step()
        out["scheduler"].step()
    assert torch.isfinite(loss), "续训后 loss 应有限: %s" % loss.item()


@case
def test_resume_with_edit_creates_fresh_group():
    """真的插了新层时，fresh 组必须出现，且预热后 lr 回升。"""
    a = mlp([8, 16, 4])
    oa = torch.optim.Adam(a.parameters(), lr=3e-3)
    train_steps(a, oa, n=20)

    b = tl.insert_after(a, "1", nn.Linear(16, 16).to(DEV))
    ob = torch.optim.Adam(b.parameters(), lr=3e-3)

    x = torch.randn(32, 8, device=DEV)
    y = torch.randn(32, 4, device=DEV)

    def loss_fn(mm, batch):
        return ((mm(batch[0]) - batch[1]) ** 2).mean()

    out = tl.resume_with_edit(a, oa, b, ob, loss_fn=loss_fn,
                              batches=[(x, y)] * 3,
                              warmup={"fresh": 30}, verbose=False)
    kinds = {g.get("plan_kind") for g in ob.param_groups}
    assert tl.KIND_FRESH in kinds, "插新层应产生 fresh 组: %s" % kinds

    fresh = [g for g in ob.param_groups if g["plan_kind"] == tl.KIND_FRESH][0]
    lr0 = fresh["lr"]
    for _ in range(31):
        out["scheduler"].step()
    assert fresh["lr"] > lr0, "预热后 lr 应上升: %s -> %s" % (lr0, fresh["lr"])


def main():
    print("torch_resume %s   设备: %s" % (tl.__version__, DEV))
    print("=" * 66)
    print("跑测试")
    print("=" * 66)
    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = len(RESULTS) - n_pass
    print()
    print("=" * 66)
    print("结果: %d 通过 / %d 失败 / 共 %d" % (n_pass, n_fail, len(RESULTS)))
    if n_fail:
        for name, s, err in RESULTS:
            if s == "FAIL":
                print("  FAILED  %s -> %s" % (name, err))
    print("=" * 66)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
