# torch-resume

**Change your model structure or loss mid-training and keep going — without starting over.**

[![CI](https://github.com/jxb01/torch-resume/actions/workflows/ci.yml/badge.svg)](https://github.com/jxb01/torch-resume/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

English | [中文](README.zh-CN.md)

---

## The problem

You train for three days, then realise your loss was wrong. Or you want to know whether a wider
hidden layer would help. The default answer is *start over* — but the only questions you actually
need answered are:

1. **Which parameters can I keep?** (the changed ones start from scratch)
2. **Is my optimizer state still valid?** (the part everyone forgets, and it always bites)

torch-resume turns those two questions into one call.

---

## Install

```bash
pip install git+https://github.com/jxb01/torch-resume.git
```

Requires PyTorch 2.0+.

---

## Quick start

```python
import torch_resume as tr

# 1) One line inside your training loop
ck = tr.Checkpoint("runs/exp1", monitor="val_loss", every=500, best=3,
                   disk_limit_gb=20)
for step, (xb, yb) in enumerate(loader):
    loss = train_step(xb, yb)
    ck.step(step, model, opt, metrics={"val_loss": loss.item()})

# 2) You changed the model -- carry on training
tr.resume_with_edit(old_model, old_opt, new_model, new_opt,
                    loss_fn=my_loss, batches=sample_batches)

# 3) Or stop and take a look
tr.pause(model, step=1200)
```

---

## What it prints

**Migration plan** — every decision is explained:

```
  0.weight        keep     shapes match
  0.bias          keep     shapes match
  2.weight        reset    overlap only 12% (below the 50% floor), (4,64) -> (32,64)
  4.weight        reset    new parameter
------------------------------------------------------------------
  summary: keep=2  reset=4
```

**Optimizer-state mismatch report** — after swapping the loss:

```
  threshold 4.0x, probed 6 batches
  2.weight     ratio= 998.345  v too large -> effective step too small (silent stall)
  0.weight     ratio= 452.579  v too large -> effective step too small (silent stall)
------------------------------------------------------------------
  summary: stall=4  oscillate=0  non-finite=0  ok=0  total=4
  advice: reset exp_avg_sq for the mismatched params and add warmup
```

---

## Four things that are not obvious

**1. The optimizer state has to come along.**
Moving weights without moving Adam's moments makes the resumed run oscillate badly.
`migrate` carries `exp_avg` / `exp_avg_sq` too, slicing them exactly like the weights.

**2. When you swap the loss, the risk is in the second moment — not the momentum.**
Adam is invariant to a *global* gradient rescale (g times k => m times k, v times k², so m/√v is
unchanged). The real issue is how long v remembers: with beta2 = 0.999 that is roughly **1000 steps**
during which it describes the *old* world.

The nastier failure is **v too large**: the effective step size becomes too small, so the parameters
look like they are moving but barely do — and you only notice hundreds of steps later.
That is why `detect_mismatch` **probes first, then changes**, instead of leaving you to guess.

**3. Slice inheritance needs an overlap floor.**
`Linear(64,4) -> Linear(64,32)` is "compatible" in the sense that every new dim is at least as large
as the old one — but the overlap is only 12.5%, and copying an old *output* layer into a new *hidden*
layer is semantically wrong. The default floor is 50%; below that it initialises from scratch instead.
Set `partial_min_ratio=0.0` to disable.

**4. Async checkpoints must clone on the training thread.**
`state_dict()` returns **references**. Without a clone, the values keep changing while the writer
flushes — you would save "the pointer of then, the value of now". That is one of the hardest bug
classes to find, so the clone happens synchronously inside `step()` and only the disk write is async.

---

## Caveats

- **`nn.Sequential` names are positional**, so they shift when you insert a layer. Migration matches by
  parameter name, so for multi-layer models prefer **named submodules** (`self.encoder = ...`) — the
  names stay stable.
- Shape-compatible does not mean semantically correct. Every `partial` decision emits a warning for you
  to confirm.
- Non-finite values (NaN/Inf) are reported as `invalid` and force a reset — they are never silently
  treated as "fine".

---

## Tests

```bash
git clone https://github.com/jxb01/torch-resume.git
cd torch-resume
pip install torch --index-url https://download.pytorch.org/whl/cpu
python tests/test_torch_resume.py     # 23 checks
python examples/demo.py               # end-to-end demo
```

Set `TORCH_RESUME_CPU=1` if you have no GPU.

Verified locally: **23/23 passing** on torch 2.14.1+cu130, RTX 5060 (sm_120).

---

## Layout

```
torch_resume/
  state.py      RNG snapshots, parameter-name maps, recursive cloning
  plan.py       structure diff -> migration plan (with overlap floor)
  migrate.py    apply the plan (weights + optimizer state)
  mismatch.py   mismatch detection (probe + per-parameter report + apply)
  checkpoint.py mandatory checkpoints
  inspect.py    pause / inspect / edit / resume_with_edit
tests/test_torch_resume.py    23 checks
examples/demo.py              end-to-end demo
```

## License

MIT
