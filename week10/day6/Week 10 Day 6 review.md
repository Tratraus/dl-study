# Week 10 Day 6 Review — Fine-tuning 与对比实验

## 完成状态

| 项目 | 状态 | 备注 |
|---|---|---|
| TODO 1: `has_hydrophobic_motif` | ✅ | 计数器法，单次遍历 O(N) |
| TODO 1: `make_motif_batch` | ✅ | 正样本插入疏水簇，负样本排除疏水 AA |
| TODO 2: `train_model` | ✅ | 冻结/FT 双模式，分组学习率 |
| TODO 3: `run_experiments` | ✅ | 三组实验 + 测试集评估 + 汇总表格 |
| Q1-Q3 | ✅ | 全部回答 |

---

## 代码结构分析

### TODO 1：数据生成

#### has_hydrophobic_motif — 计数器法

```python
def has_hydrophobic_motif(seq: list[int]) -> bool:
    count = 0
    for token in seq:
        count = count + 1 if token in HYDROPHOBIC_IDS else 0
        if count >= MOTIF_LEN:
            return True
    return False
```

**实现评价：** 比滑动窗口更优——单次遍历 O(N)，无需回看。`count` 变量追踪连续疏水长度，非疏水立即归零。简洁高效。

#### make_motif_batch

```python
# 正样本：随机序列 + 在随机位置强制插入 3 个连续疏水 AA
seq = random.choices(AA_IDS, k=seq_len)
insert_pos = random.randint(0, seq_len - MOTIF_LEN)
for j in range(MOTIF_LEN):
    seq[insert_pos + j] = random.choice(list(HYDROPHOBIC_IDS))

# 负样本：疏水 AA 权重 = 0，完全不含疏水 AA
weights_neg = [0 if i in HYDROPHOBIC_IDS else 1 for i in AA_IDS]
neg_seqs = [random.choices(AA_IDS, weights=weights_neg, k=seq_len) ...]
```

**数据流：**
```
正样本: 随机 AA 序列 + 强制插入 [H, H, H] 片段 → has_hydrophobic_motif = True
负样本: 完全不含疏水 AA 的序列 → has_hydrophobic_motif = False
拼接 + shuffle → (src, labels)
```

**信号分析：**
- 正样本：至少有 3 个连续疏水 AA（其余位置随机）
- 负样本：**完全没有**疏水 AA（12 种亲水 AA 组成）
- 信号非常清晰——负样本连单个疏水 AA 都没有，判断难度低

---

### TODO 2：train_model — 双模式训练函数

#### 冻结模式（finetune=False）

```python
for param in encoder.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
```

- encoder 参数冻结，不参与梯度更新
- 提取嵌入时用 `torch.no_grad()`
- optimizer 只管分类头

#### Fine-tuning 模式（finetune=True）

```python
for param in encoder.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam([
    {'params': encoder.parameters(),    'lr': 1e-4},
    {'params': classifier.parameters(), 'lr': 1e-3},
])
```

- encoder 解冻，参与梯度更新
- **分组学习率**：encoder 1e-4（小，防灾难性遗忘），classifier 1e-3（大，快收敛）
- 提取嵌入时**不用** `torch.no_grad()`

**关键区别对照：**

| | 冻结 | Fine-tuning |
|---|---|---|
| `requires_grad` | False | True |
| encoder 模式 | `.eval()` | `.train()` |
| `torch.no_grad()` | ✅ | ❌ |
| optimizer | 只有 classifier | encoder + classifier |
| 学习率 | 1e-3 | encoder 1e-4 + classifier 1e-3 |

---

### TODO 3：run_experiments — 三组对比实验

```python
# A: 随机 Encoder + 冻结（不加载 checkpoint）
encoder_A = ProteinBERT().to(device)

# B: 预训练 Encoder + 冻结
encoder_B.load_state_dict(ckpt['model_state_dict'])

# C: 预训练 Encoder + Fine-tuning
encoder_C.load_state_dict(ckpt['model_state_dict'])
train_model(encoder_C, finetune=True)
```

测试集评估：`test_acc` 函数在 `__main__` 内定义为闭包，复用 `make_motif_batch` 生成独立测试数据。

---

## 训练结果分析

### 输出

```
实验 A：随机 Encoder + 冻结
  Step    0 | loss: 0.6924 | acc: 0.5000
  Step   50 | loss: 0.0864 | acc: 1.0000    ← 50 步收敛

实验 B：预训练 Encoder + 冻结
  Step    0 | loss: 0.7323 | acc: 0.5000
  Step   50 | loss: 0.1972 | acc: 0.9844    ← 略慢
  Step  100 | loss: 0.0514 | acc: 1.0000

实验 C：预训练 Encoder + Fine-tuning
  Step    0 | loss: 0.7094 | acc: 0.5000
  Step   50 | loss: 0.0024 | acc: 1.0000    ← 最快收敛，loss 极低
```

### 测试集对比

| 实验 | 配置 | 测试准确率 |
|---|---|---|
| A | 随机 Encoder + 冻结 | **1.0000** |
| B | 预训练 Encoder + 冻结 | 0.9992 |
| C | 预训练 Encoder + FT | **1.0000** |

### 深度分析：为什么三组差异这么小？

**核心原因：任务太简单，预训练优势无法体现。**

1. **负样本完全没有疏水 AA** — 模型只需学会"有没有疏水 AA"，不需要理解局部上下文
2. **随机 Encoder 也能轻松解决** — 12 种亲水 AA vs 8 种疏水 AA，Embedding 层就能区分
3. **预训练 Encoder 学到的特征对这个任务没有额外帮助** — MLM 预训练在随机序列上进行，没有学到有意义的疏水模式

**实验 A > 实验 B 的反直觉现象：**

随机 Encoder (1.0000) 略优于预训练 Encoder (0.9992)。可能原因：
- 预训练 Encoder 的 Embedding 空间经过优化，不同 AA 的表示更"紧凑"，反而不如随机初始化时那么容易线性可分
- 冻结后分类头只能利用现有表示，不能调整
- 差距极小 (0.08%)，可能是随机波动

**实验 C 收敛最快：**

Fine-tuning 在 step 50 就 loss 降到 0.0024，因为 encoder 参数可以调整以适应任务，而冻结模式下分类头需要自己学习如何利用固定表示。

---

## 关键知识点

1. **分组学习率语法** — `Adam([{'params': ..., 'lr': ...}, ...])` 为不同参数组设置不同学习率
2. **灾难性遗忘** — Fine-tuning 时 encoder 用小学习率 (1e-4) 防止破坏预训练知识
3. **实验设计** — A vs B 验证预训练价值，B vs C 验证 Fine-tuning 价值
4. **任务难度决定实验结论** — 任务太简单时，所有方法都能达到上限，无法区分优劣
5. **冻结 vs Fine-tuning 的权衡** — 冻结更快更稳（不会破坏预训练特征），Fine-tuning 更灵活但需要小心学习率

---

## 踩坑易错点

| 问题 | 原因 | 修正 |
|---|---|---|
| Fine-tuning 时 encoder 用大学习率 | 灾难性遗忘，破坏预训练知识 | encoder lr=1e-4, classifier lr=1e-3 |
| Fine-tuning 时忘记 `encoder.train()` | Dropout/BatchNorm 行为错误 | 解冻后切换到 train 模式 |
| 冻结时忘记 `torch.no_grad()` | 构建不必要的计算图，浪费显存 | 冻结模式提取嵌入时包在 no_grad 里 |
| 任务太简单导致实验无区分度 | 负样本完全不含疏水 AA | 应设计更难的任务（比如需要 motif 位置信息） |

---

## Q1-Q3 回答

**Q1**：Fine-tuning 时为什么 Encoder 用 lr=1e-4，分类头用 lr=1e-3？如果两者都用 1e-3 会有什么风险？

> Encoder 已经经过预训练，参数已经在一个较好的区域，如果用过大的学习率（如 1e-3）可能会导致预训练的知识被破坏（catastrophic forgetting），反而降低性能。分类头是新初始化的，需要更大的学习率来快速收敛。

✅ 完全正确。补充：实际中如果两者都用 1e-3，encoder 的参数会被"拉"向当前任务的最优解，远离预训练学到的通用特征。在数据量少时尤其危险——过拟合当前任务，丧失泛化能力。

**Q2**：实验 A（随机 Encoder）和实验 B（预训练 Encoder）的准确率差距是多少？你如何解释这个差距（或者没有差距）？

> 实验 A 和 B 的测试集准确率都非常高（A: 1.0000, B: 0.9992），说明在这个简单的 Motif 检测任务上，随机初始化的 Encoder 也能学到足够的信息来区分正负样本。预训练的 Encoder 在这个任务上没有明显优势，可能是因为 Motif 检测任务相对简单，随机 Encoder 也能通过训练快速适应。

✅ 分析正确。补充根本原因：这个任务的信号太强（负样本完全没有疏水 AA），不需要预训练就能解决。要体现预训练价值，需要设计更难的任务——比如区分"疏水 AA 随机散布"vs"疏水 AA 形成连续 motif"，两者疏水比例相同但排列不同。

**Q3**：实验 B（冻结）和实验 C（Fine-tuning）相比，哪个更好？为什么 Fine-tuning 不一定总是更好？

> 实验 C 的测试集准确率略高于 B（C: 1.0000, B: 0.9992），但差距非常小。在这个任务上，Fine-tuning 没有显著提升性能，可能是因为预训练的 Encoder 已经足够好，冻结它也能达到接近最优的结果。

✅ 正确。补充：Fine-tuning 不一定更好的三个原因：
1. **数据量少时容易过拟合** — encoder 参数多，小数据集上学到的可能是噪声
2. **灾难性遗忘** — 学习率控制不好会破坏预训练特征
3. **任务简单时无必要** — 冻结已经能达到上限，微调只是浪费计算
