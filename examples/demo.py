# -*- coding: utf-8 -*-
"""torch_resume 演示：训练到一半改结构、改 loss，不用从头再来。

  python examples/demo.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

import torch_resume as tl

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def banner(t):
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def build(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(20, 64), nn.ReLU(),
        nn.Linear(64, 4),
    ).to(DEV)


def make_data(n=512, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 20, generator=g).to(DEV)
    # 目标函数：让模型真的能学到东西
    y = torch.stack([x[:, 0] * 2 + x[:, 1], x[:, 2] - x[:, 3],
                     (x[:, 4] ** 2).clamp(max=3), x[:, 5:10].sum(1) * 0.3], dim=1)
    return x, y


def loss_fn(m, batch):
    xb, yb = batch
    return ((m(xb) - yb) ** 2).mean()


def train(model, opt, x, y, steps, tag="", log_every=None):
    model.train()
    last = None
    for i in range(steps):
        loss = loss_fn(model, (x, y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
        if log_every and (i + 1) % log_every == 0:
            print("  [%s] step %4d   loss %.6f" % (tag, i + 1, last))
    return last


def main():
    print("torch_resume %s   设备: %s" % (tl.__version__, DEV))
    x, y = make_data()
    ckdir = tempfile.mkdtemp(prefix="tl_demo_")
    try:
        # ---------------------------------------------------------- 1
        banner("1. 正常训练 300 步（带强制检查点）")
        m1 = build()
        o1 = torch.optim.Adam(m1.parameters(), lr=3e-3)
        ck = tl.Checkpoint(ckdir, monitor="loss", every=100, best=2,
                           keep_last=2, disk_limit_gb=1.0, verbose=False)
        for step in range(1, 301):
            l = train(m1, o1, x, y, 1)
            ck.step(step, m1, o1, metrics={"loss": l})
            if step % 100 == 0:
                print("  step %3d   loss %.6f" % (step, l))
        print("  检查点目录:")
        for f in sorted(os.listdir(ckdir)):
            print("    " + f)

        # ---------------------------------------------------------- 2
        banner("2. 暂停：查看每个参数的统计")
        print(tl.param_table(m1))
        tl.print_params(m1)

        # ---------------------------------------------------------- 3
        banner("3a. 改结构：把隐藏层加宽 64 -> 128，改完直接续训")
        m2 = nn.Sequential(
            nn.Linear(20, 128), nn.ReLU(),
            nn.Linear(128, 4),
        ).to(DEV)
        o2 = torch.optim.Adam(m2.parameters(), lr=3e-3)

        out = tl.resume_with_edit(m1, o1, m2, o2,
                                  loss_fn=loss_fn, batches=[(x, y)] * 4,
                                  threshold=4.0, policy="threshold")
        # 验证第一层确实继承了（必须在续训之前查，训练之后当然会变）
        # 加宽场景下形状从 (64,20) 变 (128,20)，所以要比较前 64 行
        w_old = m1[0].weight.detach()
        w_new = m2[0].weight.detach()
        same = torch.equal(w_new[: w_old.shape[0]], w_old)
        print("  第一层已训练的 %d 行是否逐位继承: %s"
              % (w_old.shape[0], "是" if same else "否"))
        assert same, "继承的行必须逐位相同"

        loss_after = train(m2, o2, x, y, 200, tag="续训", log_every=100)
        print("  续训 200 步后 loss = %.6f" % loss_after)
        assert torch.isfinite(torch.tensor(loss_after)), "续训不应出现 NaN"

        banner("3b. 对照：加一层（重叠太低 -> 宁可从零初始化）")
        m2b = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4),
        ).to(DEV)
        old_sd = {k: v.detach().cpu() for k, v in m1.state_dict().items()}
        print(tl.diff(m2b, old_sd).report())

        # ---------------------------------------------------------- 4
        banner("4. 换 loss（尺度剧变）→ 先试算，再决定怎么处理")
        m3 = build(seed=1)
        o3 = torch.optim.Adam(m3.parameters(), lr=3e-3)
        train(m3, o3, x, y, 300)
        print("  已训练 300 步，现在换成 loss * 0.02")

        def new_loss(m, batch):
            xb, yb = batch
            return ((m(xb) - yb) ** 2).mean() * 0.02

        o4 = torch.optim.Adam(m3.parameters(), lr=3e-3)
        tl.migrate(m3, o4, m3, o3, verbose=False)
        rep = tl.detect_mismatch(m3, o4, new_loss, [(x, y)] * 6, threshold=4.0)
        print(rep.report())
        n = rep.apply(o4, policy="threshold")
        print("  [处理] 策略=threshold，重置了 %d 个参数的 exp_avg_sq" % n)
        loss_after2 = train(m3, o4, x, y, 300, tag="新loss", log_every=150)
        print("  换 loss 后续训 300 步 loss = %.6f（未出现 NaN）" % loss_after2)
        assert torch.isfinite(torch.tensor(loss_after2))

        # ---------------------------------------------------------- 5
        banner("5. 崩溃恢复")
        hint = ck.resume_hint()
        print("  resume_hint: %s" % hint)
        m4 = build(seed=42)
        o5 = torch.optim.Adam(m4.parameters(), lr=3e-3)
        payload = ck.load(ck.latest_path(), m4, o5)
        print("  已从 step %s 恢复" % payload["step"])
        print("  恢复后第一层是否与保存时一致:",
              torch.equal(m4[0].weight.detach(), m1[0].weight.detach()))

        banner("全部通过 —— 训练到一半改结构/改 loss，没有从头再来")
    finally:
        shutil.rmtree(ckdir, ignore_errors=True)


if __name__ == "__main__":
    main()
