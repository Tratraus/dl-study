# 深度学习实验随机种子复现性排查报告

> 案例来源：Week 13 Day 5 蛋白质亚细胞定位分类任务
> 排查周期：3 轮迭代（Baseline v2 → v4 → v5）
> 最终结果：两次独立运行 **bit-exact 完全一致**

---

## 一、问题现象

在"固定了 `torch.manual_seed(42)`"的前提下，同一套 Baseline 代码在不同场景下跑出了**四个不同的结果**：

| 版本 | Test Acc | Macro F1 | 备注 |
| :--- | :---: | :---: | :--- |
| Day1 原始 | 0.6124 | 0.4527 | 完全未加种子 |
| Day4 校正 | 0.6331 | 0.4860 | 加了种子，但结果仍漂移 |
| v2 | 0.6378 | 0.4890 | 又加了一轮修复，依然不对 |
| v4 | 0.6386 | 0.4782 | 差异缩小到 0.01 级别，但没归零 |
| **v5（最终）** | **0.6481** | **0.4643** | ✅ 归零，两次独立运行 bit-exact |

**最危险的地方在于**：每一轮的差异都在"看起来像是合理的训练方差范围内"（±1~3个百分点），非常容易被误判为"这就是随机性正常波动，不是bug"，从而放弃排查——但事实证明，这背后是三个真实存在、可修复的代码问题。

---

## 二、根因排查：三个独立叠加的 Bug

### Bug ① GPU 并行算子的非确定性

**现象**：即使 `manual_seed` 完全固定，同一段代码在 GPU 上跑两次，结果依然有细微差异。

**原理**：GPU 上很多归约类算子（如 `scatter_add`、某些卷积实现）为了追求速度，会用非确定性的并行调度方式，导致浮点数累加顺序在每次运行时不同，从而产生微小误差。这种误差会在多层网络、多个 epoch 的训练中被逐渐放大。

**修复代码**：
```python
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.use_deterministic_algorithms(True)
```
⚠️ 这两行必须放在脚本**最开头**，在 `import torch` 之后立刻设置，越早越好。

---

### Bug ② 关键函数内部缺失局部 `set_seed` 调用

**现象**：即使脚本最开头调用了一次 `set_seed(42)`，只要中间跑过其他实验（比如 Frozen/Fine-tune/Probing 等热身实验），Baseline 函数内部的模型初始化用到的随机数生成器状态，已经被前面的实验"消耗"过了。

**原理**：`torch.manual_seed(42)` 只在调用那一刻重置一次随机数生成器。之后每一次 `nn.Linear()`、`nn.Dropout()`等带随机初始化的操作，都会消耗生成器状态并让它继续往后走。**如果脚本里在 Baseline 之前跑过任何别的实验，Baseline 的模型初始化权重就已经不再是"seed=42下的第一次初始化"了**。

**修复代码**：
```python
def run_baseline(train_loader, val_loader, test_loader,
                 tokenizer, num_classes, epochs=20, seed=42):
    set_seed(seed)   # 关键：函数入口显式重置，不依赖脚本开头那一次
    model = ProteinClassifier(...)
    ...
```
**教训**：**任何一个需要复现的实验单元（函数/模块），都应该在入口处独立重置种子，不能依赖"脚本开头调用过一次就够了"这个假设。**

---

### Bug ③ DataLoader 的 `generator` 状态被跨组共享/消耗（本案例中最隐蔽的一环）

**现象**：即使 Bug①②都修复了，A/B/C 三组实验（或 Baseline 与其他实验）之间，训练数据的 shuffle 顺序依然不同。

**原理**：`DataLoader(shuffle=True)` 内部依赖一个随机数生成器来决定每个 epoch 的样本顺序。如果多个实验**复用同一个 `train_loader` 对象**，那么：
- 第一个实验跑完后，`train_loader` 内部的生成器状态已经往后推进了 N 步（N = epoch数 × 每epoch的shuffle次数）
- 第二个实验开始时，看到的"随机顺序"根本不是从 `manual_seed(42)` 的起点开始的，而是接着第一个实验消耗完的地方继续

这解释了为什么**模型初始化完全相同**，但最终结果依然不同——**两组模型看到的训练数据顺序不一样，训练轨迹自然分叉**。

**修复代码**：
```python
# 错误写法：直接复用外部传入的 train_loader
def run_experiment(train_loader, ...):
    for epoch in range(epochs):
        for batch in train_loader:   # ← generator状态可能已被污染
            ...

# 正确写法：每个实验单元内部，用全新 generator 重建 DataLoader
def run_experiment(train_loader, ..., seed=42):
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        collate_fn=train_loader.collate_fn,
        generator=torch.Generator().manual_seed(seed)   # 关键：全新独立生成器
    )
    for epoch in range(epochs):
        for batch in train_loader:
            ...
```

---

## 三、三个 Bug 的叠加效应

三个问题**单独修复都不够**，必须同时修复才能归零：

| 只修复 | 结果 |
| :--- | :--- |
| 只修①（GPU确定性算子） | 结果仍漂移（②③依然存在） |
| 只修①② | 差异从大幅缩小到约0.01级别，但仍未归零（③依然存在） |
| **①②③全部修复** | **两次独立运行 bit-exact 完全一致** |

这也是为什么中间几轮（v2/v4）看起来"越修越接近但总是差一点"——**这是多因叠加问题的典型特征：每修一个因子，残余误差会显著缩小，但只要还有一个没修，就永远无法归零。**

---

## 四、通用排查清单（以后遇到"结果对不上"时直接照此排查）

遇到"固定了种子但结果依然不可复现"的问题时，按以下顺序逐项排查：

- [ ] **硬件/框架层**：是否设置了 `torch.use_deterministic_algorithms(True)` 和 `CUBLAS_WORKSPACE_CONFIG`？（PyTorch + CUDA 场景下几乎必做）
- [ ] **全局种子**：`torch.manual_seed`、`torch.cuda.manual_seed_all`、`numpy.random.seed`、`random.seed` 是否都设置了？（只设一个框架的种子是最常见的漏洞）
- [ ] **函数/模块入口**：每一个需要独立复现的实验单元（哪怕在同一个脚本里跑多组实验），是否都在**入口处**重新调用了 `set_seed()`？不能假设"脚本开头调用一次就够"
- [ ] **DataLoader 的 generator**：是否有多个实验/多个函数**共享同一个 DataLoader 对象**？如果有，其内部 shuffle 用的 `generator` 状态会被跨组消耗污染——每组独立实验前需要用全新 `torch.Generator().manual_seed(seed)` 重建 DataLoader
- [ ] **模型/数据加载过程本身**：像 `load_esm2()` 这种加载预训练权重的函数，内部是否隐藏了未固定种子的随机初始化（比如新增的分类头、pooler层）？
- [ ] **DataLoader 的 `num_workers`**：如果 `num_workers > 0`，每个 worker 进程也需要设置 `worker_init_fn` 固定种子，否则多进程下的随机性也无法复现（本案例未涉及，但常见踩坑点）
- [ ] **验证方法**：不要只跑一次就下结论。**用同一套代码独立跑两次，对比结果是否 bit-exact 完全一致**（不是"差不多"，而是小数点后所有位都一样）。只有 bit-exact 才能证明复现链路彻底闭环。

---

## 五、核心教训（一句话总结）

> **"固定了 manual_seed" ≠ "结果可复现"**。真正的可复现性需要同时保证：硬件算子确定性、每个独立实验单元的种子重置、以及所有涉及随机采样的对象（尤其是 DataLoader）不被跨实验共享/污染。任何一环缺失，都会产生"看起来像训练方差、实际是代码bug"的伪随机性，极易误判为不可控问题而放弃排查。

---

## 六、验证方法模板（可直接复用）

```python
# 在任何"固定种子"的实验前后，加这段验证代码
def verify_reproducibility(run_fn, *args, n_runs=2, **kwargs):
    results = []
    for i in range(n_runs):
        acc, f1 = run_fn(*args, **kwargs)
        results.append((acc, f1))
        print(f"Run {i+1}: Acc={acc:.10f} | F1={f1:.10f}")

    is_exact = all(r == results[0] for r in results)
    print(f"\n{'✅ Bit-exact 完全一致' if is_exact else '❌ 存在差异，复现性未闭环'}")
    return is_exact
```