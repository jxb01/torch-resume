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

## 六个能力

| 模块 | 做什么 |
|---|---|
| `Checkpoint` | 强制保存：最优 + 阶段性、原子写、异步、保留策略、磁盘限额、崩溃恢复 |
| `diff` / `migrate` | 改结构后继承权重与**优化器状态**（含 Adam 动量） |
| `detect_mismatch` | 换 loss 后**先试算再改**：逐参数比对 sqrt(v) 与新梯度，报告并处理 |
| `Inspector` / `pause` | 暂停、看每个参数的统计、直接改权重、冻结/解冻 |
| `widen` / `replace_module` / `insert_after` / `duplicate_layer` | 模型手术：改结构不用重写模型 |
| `retag_groups` / `PlanWarmup` | 按参数类别分组，对全新/失配的参数自动预热 |
| `ModelCache` | 结构缓存：改之前存一份，改坏了能变回去 |
| `resume_with_edit` | 上面几件事的一条命令版本 |

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

# 2) 把隐藏层加宽 —— 一行；已经训好的模型原样不动
new_model = tr.widen(model, "0", 128)
new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)

# 3) 接着训：权重和优化器状态一起继承，再按类别预热
out = tr.resume_with_edit(
    model, opt, new_model, new_opt,
    loss_fn=my_loss, batches=sample_batches,
    warmup={"fresh": 500, "mismatched": 200},
    lr_by_kind={"fresh": 2.0})

for step, (xb, yb) in enumerate(loader):
    ...
    out["scheduler"].step()        # 在 optimizer.step() 之后

# 4) 或者暂停进去看看
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

## 模型手术

改一层不用重写整个模型。

```python
new = tr.widen(model, "0", 128)                    # 加宽 + 同步加宽下游 + 中间夹的 norm
new = tr.replace_module(model, "encoder.layer.3", my_layer)
new = tr.insert_after(model, "blocks.2", blk)      # 下标位移，重命名映射自动算好
new = tr.duplicate_layer(model, "blocks.1", n=2)   # 扩深；新份直接复制原层权重
```

每次都返回**新模型**（旧的留着做迁移对比），并记录做过什么：

```
模型手术记录
  widen 0: 64 -> 128（下游 2）
  显式继承: 4 项
```

这份记录让迁移能把两边对上 —— **没有记录就不猜**。

---

## 自动 warmup

以前 `detect_mismatch` 只是"建议加 warmup"，现在库自己执行：

```python
out = tr.resume_with_edit(..., warmup={"fresh": 500, "mismatched": 200},
                          lr_by_kind={"fresh": 2.0})
```

参数按"实际发生了什么"重新分组，每组有自己的调度：

```
参数分组
  fresh        lr=6.000e-03   4 个   权重从零初始化
  mismatched   lr=3.000e-03   3 个   优化器状态被重置
  kept         lr=3.000e-03   1 个   完全继承

warmup 计划
  fresh        lr=6.000e-04   4 个   预热 500 步
  mismatched   lr=3.000e-04   3 个   预热 200 步
  kept         lr=3.000e-03   1 个   不预热
```

`retag_groups` **不会动 `optimizer.state`**（它按参数对象索引），所以刚迁移过来的动量还在。

---

## 改坏了就变回去

改结构是一个**假设**，有时它是错的。而 `state_dict` 救不了你 —— **它只存权重，不存结构**。
所以缓存里存三样：结构、权重、优化器状态。

```python
cache = tr.ModelCache("runs/exp1/model_cache")

cache.save(model, opt, tag="good", note="加宽之前那一版")

bad = tr.widen(model, "0", 128)
... 训一会，发现更差 ...

model, opt, info = cache.restore("good", model, opt)   # 回到原来那一版
```

```
模型结构缓存  runs/exp1/model_cache
========================================================================
  good      10-06 22:42:01   1604 参数  step -   val_loss=0.31  # 加宽之前
  wide      10-06 23:05:44   2884 参数  step -   val_loss=0.48  # 候选
------------------------------------------------------------------------
  共 2 份，占用 0.4 MB
```

**三级恢复，按可靠性排：**

| 模式 | 什么时候 | 做什么 |
|---|---|---|
| **inplace** | 目标结构一致 | `load_state_dict` —— 最快最稳，推荐路径 |
| **rebuild** | 结构不一致 | 用存档里 pickle 的模型对象重建（要求那个类可反序列化） |
| **拒绝** | 连对象也反序列化不了 | 报错并打印**逐参数的差异** —— **不猜** |

和 `resume_with_edit` 一起用，它会在动手之前自动替你存档：

```python
out = tr.resume_with_edit(old, old_opt, new, new_opt, cache=cache, ...)
# out["snapshot"] 就是自动存下的回滚点
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
python tests/test_torch_resume.py     # 43 项
python examples/demo.py               # 端到端演示
```

没有 GPU 时加 `TORCH_RESUME_CPU=1` 环境变量。

---

## 文件

```
torch_resume/
  _io.py         编码安全的打印（Windows 控制台不是 UTF-8）
  _store.py      原子写盘，检查点与缓存共用
  state.py       RNG 快照、参数名映射、递归克隆
  plan.py        结构对比 -> 迁移计划（重叠比例门槛 + 手术记录）
  migrate.py     执行迁移（权重 + 优化器状态）
  mismatch.py    失配检测（试算 + 逐参数报告 + 处理）
  groups.py      按类别重新分组 + PlanWarmup 调度器
  surgery.py     widen / replace_module / insert_after / duplicate_layer
  cache.py       结构 + 权重 + 优化器状态的存档与回滚
  checkpoint.py  强制检查点
  inspect.py     暂停 / 查看 / 修改 / resume_with_edit
tests/test_torch_resume.py     35 项测试
examples/demo.py               端到端演示
```

## License

MIT
