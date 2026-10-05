# torch-resume

**让训练不用从头再来。** 一个 PyTorch 的训练干预工具箱。

[English](README.md) | 中文

只做一件事：**训练到一半改结构、改 loss，接着训，不重来。**

---

## 它解决什么

训练三天发现 loss 选错了，或者想知道把隐藏层加宽会不会更好 —— 默认做法是从头再来。
但你要问的其实只有两个问题：

1. **哪些参数可以保留？**（改动的部分从零开始）
2. **优化器状态还适用吗？**（这个最容易漏，漏了必然出问题）

torch-resume 把这两个问题变成一次调用。

---

## 安装

```bash
pip install git+https://github.com/jxb01/torch-resume.git
```

需要 PyTorch 2.0+。如果你用 CUDA 版：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/jxb01/torch-resume.git
```

---

## 五个能力

| 模块 | 做什么 |
|---|---|
| `Checkpoint` | 强制保存：最优 + 阶段性、原子写、异步、保留策略、磁盘限额、崩溃恢复 |
| `diff` / `migrate` | 改结构后继承权重与**优化器状态**（含 Adam 动量） |
| `detect_mismatch` | 换 loss 后**先试算再改**：逐参数比对 sqrt(v) 与新梯度，报告并处理 |
| `Inspector` / `pause` | 暂停、看每个参数的统计、直接改权重、冻结/解冻 |
| `resume_with_edit` | 上面三件事的一条命令版本 |

---

## 快速开始

```python
import torch_resume as tr

# 1) 训练循环里加一行
ck = tr.Checkpoint("runs/exp1", monitor="val_loss", every=500, best=3,
                   disk_limit_gb=20)
for step, (xb, yb) in enumerate(loader):
    loss = train_step(xb, yb)
    ck.step(step, model, opt, metrics={"val_loss": loss.item()})

# 2) 改了模型结构，接着训
tr.resume_with_edit(old_model, old_opt, new_model, new_opt,
                    loss_fn=my_loss, batches=sample_batches)

# 3) 或者暂停进去看看
tr.pause(model, step=1200)
```

---

## 它会打印什么

**迁移计划**（每个决定都能解释）：

```
  0.weight        keep     形状一致
  0.bias          keep     形状一致
  2.weight        reset    从零初始化（重叠仅 12%（低于 50% 门槛）(4,64) -> (32,64)）
  4.weight        reset    从零初始化（新增参数）
------------------------------------------------------------------
  汇总: keep=2  reset=4
```

**失配检测**（换 loss 之后）：

```
  阈值 4.0x，试算 6 个 batch
  2.weight     ratio= 998.345  v 偏大 -> 有效步长偏小（会停滞，难发现）
  0.weight     ratio= 452.579  v 偏大 -> 有效步长偏小（会停滞，难发现）
------------------------------------------------------------------
  汇总: 停滞=4  震荡=0  非有限=0  正常=0  共=4
  建议: 对失配参数重置 exp_avg_sq，并加 warmup
```

---

## 四个不显然但重要的设计点

**1. 优化器状态必须一起迁。**
只搬权重、不搬 Adam 的动量，续训会剧烈震荡。`migrate` 连 `exp_avg` / `exp_avg_sq`
一起搬，部分继承的参数按**同样的切片**迁动量。

**2. 换 loss 时，风险在二阶矩而不在动量。**
Adam 对梯度的**整体缩放不敏感**（g 乘 k，m 乘 k，v 乘 k²，m/√v 不变），
所以给 loss 乘个系数自动就对了。真正的问题是 v 的记忆长度：
beta2 = 0.999 时约 **1000 步**，期间它是"旧世界的统计量"。

最阴险的是 **v 偏大**：有效步长偏小，参数看起来在更新、实际几乎没动，往往几百步后才发现。
所以 `detect_mismatch` **先试算再改**，而不是让你去猜。

**3. 切片继承必须有重叠比例门槛。**
`Linear(64,4) -> Linear(64,32)` 在"每维都不小于"的意义上是兼容的，
但重叠只有 12.5% —— 把旧的输出层搬进新的隐藏层是语义错误。
默认门槛 50%，低于就宁可从零初始化（`partial_min_ratio=0.0` 可关掉）。

**4. 异步保存必须在训练线程上 clone 快照。**
`state_dict()` 返回的是**引用**，不 clone 的话异步写盘期间值会被训练改掉 ——
存下来的是"当时的指针、之后的值"。这是最难发现的一类 bug，
所以克隆在 `step()` 里同步发生，写盘才异步。

---

## 注意

- **`nn.Sequential` 的位置命名在插入层之后会错位。** 迁移按参数名匹配，
  所以多层模型建议用**命名子模块**（`self.encoder = ...`），名字稳定得多。
- 形状兼容不代表语义正确。`partial` 一定会给警告，请人工确认。
- 非有限值（NaN/Inf）在失配检测里会被标成 `invalid` 并要求重置，
  不会被当成"正常"静默放过。

---

## 运行测试

```bash
git clone https://github.com/jxb01/torch-resume.git
cd torch-resume
pip install torch --index-url https://download.pytorch.org/whl/cpu
python tests/test_torch_resume.py     # 23 项
python examples/demo.py               # 端到端演示
```

没有 GPU 时加 `TORCH_RESUME_CPU=1` 环境变量。

---

## 文件

```
torch_resume/
  state.py       RNG 快照、参数名映射、递归克隆
  plan.py        结构对比 -> 迁移计划（含重叠比例门槛）
  migrate.py     执行迁移（权重 + 优化器状态）
  mismatch.py    失配检测（试算 + 逐参数报告 + 处理）
  checkpoint.py  强制检查点
  inspect.py     暂停 / 查看 / 修改 / resume_with_edit
tests/test_torch_resume.py     23 项测试
examples/demo.py               端到端演示
```

## License

MIT
