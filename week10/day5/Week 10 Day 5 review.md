# Week 10 Day 5 Review — 下游分类任务

## 完成状态

| 项目 | 状态 | 备注 |
|---|---|---|
| TODO 1: `make_classification_batch` | ✅ | 权重采样 + 打乱 |
| TODO 2: `ClassificationHead` | ✅ | nn.Sequential 两层 MLP |
| TODO 3: `train_classifier` | ✅ | 冻结 encoder + 训练分类头 |
| TODO 4: `evaluate` | ✅ | 多批次统计准确率 |
| Q1-Q3 | ✅ | 全部回答 |

---

## 代码结构分析

### TODO 1：make_classification_batch

```python
weights_pos = [5 if i in KR_IDS else 1 for i in AA_IDS]  # K/R 权重 5
weights_neg = [0 if i in KR_IDS else 1 for i in AA_IDS]  # K/R 权重 0

pos_seqs = [random.choices(AA_IDS, weights=weights_pos, k=seq_len) for _ in range(half)]
neg_seqs = [random.choices(AA_IDS, weights=weights_neg, k=seq_len) for _ in range(half)]
```

**数据流：**
```
AA_IDS: 20 种氨基酸 id
weights_pos: [1, 1, 1, 1, 1, 5, 1, 1, 5, 1, ...]  ← K/R 位置为 5
weights_neg: [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, ...]  ← K/R 位置为 0

random.choices(AA_IDS, weights=..., k=30)
→ 按权重采样 30 个氨基酸 id
→ 正样本: K/R 出现概率高 (~22%)
→ 负样本: K/R 完全不出现 (0%)

拼接 + shuffle → (src, labels)
src:    (64, 30) long
labels: (64,)    long
```

**信号强度分析：**
- 正样本 K/R 比例 ≈ 5/(18+5+5) × 2 ≈ 22%，刚好超过 KR_THRESHOLD=0.2
- 负样本 K/R 比例 = 0%，远低于阈值
- 信号非常清晰，分类难度低

---

### TODO 2：ClassificationHead

```python
self.net = nn.Sequential(
    nn.Linear(d_model, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, num_classes)
)
```

**参数量：**
- Linear(128, 64): 128×64 + 64 = 8,256
- Linear(64, 2): 64×2 + 2 = 130
- 总计: 8,386

只有 encoder 参数量 (469,657) 的 1.8%，冻结 encoder 后可训练参数极少。

---

### TODO 3：train_classifier

```python
# 冻结 encoder
for param in encoder.parameters():
    param.requires_grad = False

# 每步
with torch.no_grad():
    embeddings = mean_pooling(encoder, src)   # (batch, 128)
logits = classifier(embeddings)               # (batch, 2)
loss = F.cross_entropy(logits, labels)
# backward + step
```

**关键设计决策：**
- `encoder.eval()` + `torch.no_grad()` → 不追踪 encoder 的计算图，节省显存
- `optimizer = Adam(classifier.parameters())` → 只更新分类头的 8,386 个参数
- `logits.argmax(dim=-1)` → dim=-1 和 dim=1 等价（batch 维度），更通用

---

### TODO 4：evaluate

```python
correct = 0
total = 0
for _ in range(num_batches):
    src, labels = make_classification_batch(batch_size, seq_len, device)
    with torch.no_grad():
        embeddings = mean_pooling(encoder, src)
        logits = classifier(embeddings)
        preds = logits.argmax(dim=-1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)
return correct / total
```

**小问题：** `device` 变量在函数内未定义，依赖外部作用域。实际运行时因为 Python 闭包可以访问 `__main__` 中的 `device`，不会报错。更规范的写法是加 `device` 参数或用 `src.device`。

---

## 训练结果分析

### 输出

```
Step    0 | loss: 0.7359 | acc: 0.5000    ← 初始随机猜测
Step   50 | loss: 0.1371 | acc: 0.9688    ← 快速学习
Step  100 | loss: 0.0272 | acc: 1.0000    ← 已经完美
Step  500 | loss: 0.0005 | acc: 1.0000    ← loss 继续压低

测试集准确率：0.9992
```

### 收敛曲线

```
loss:   0.74 → 0.14 → 0.03 → 0.0005   (1500× 下降)
acc:    0.50 → 0.97 → 1.00 → 1.00     (50 步内收敛)
```

### 为什么这么快就收敛了？

1. **任务简单**：K/R 频率是一个非常直接的信号，不需要复杂的序列模式
2. **信号清晰**：负样本完全不含 K/R（权重=0），正样本 K/R 明显偏多（权重=5），几乎没有模糊地带
3. **预训练嵌入质量高**：即使 MLM loss 只降到 2.88，encoder 已经学会了区分不同氨基酸的表示
4. **分类头容量小**：只有 8,386 个参数，不容易过拟合

### 测试集 0.9992 vs 训练集 1.0000

差距极小 (0.08%)，说明：
- 没有过拟合
- 数据生成规则简单且一致，泛化不是问题
- 如果换更复杂的分类规则（比如 motif 组合），差距可能会拉大

---

## 关键知识点

1. **冻结 Encoder 的意义** — 只训练分类头，验证预训练嵌入是否包含下游任务所需的信息。如果冻结后效果就好，说明预训练成功
2. **torch.no_grad() 的作用** — 不只是节省显存，更重要的是不构建计算图，反向传播不会流过 encoder
3. **requires_grad=False vs torch.no_grad()** — 前者冻结参数（不会被更新），后者关闭梯度追踪（节省内存）。两者通常一起用
4. **权重采样生成数据** — `random.choices(weights=...)` 是构造有偏数据的简单方法，比手写 if-else 更优雅
5. **argmax(dim=-1)** — dim=-1 表示最后一个维度，在 2D tensor 中等价于 dim=1，但更通用

---

## 踩坑易错点

| 问题 | 原因 | 修正 |
|---|---|---|
| optimizer 包含 encoder 参数 | 写成 `model.parameters()` 而非 `classifier.parameters()` | 冻结后只传 classifier 的参数 |
| 忘记 torch.no_grad() | encoder 虽然冻结但仍构建计算图，浪费显存 | 提取嵌入时包在 no_grad 里 |
| evaluate 中 device 未定义 | 依赖闭包访问外部变量 | 加 device 参数或用 `src.device` |

---

## Q1-Q3 回答

**Q1**：冻结 Encoder 后，如果误写成 `optimizer = torch.optim.Adam(model.parameters(), lr=lr)`，会发生什么？

> 会导致整个模型（包括 encoder）都参与优化，虽然 encoder 的参数 requires_grad=False，但 optimizer 仍然会尝试更新它们。这可能会浪费计算资源，并且可能会引入不必要的噪声，影响分类头的训练效果。

✅ 部分正确。补充：`requires_grad=False` 的参数**不会被更新**（梯度为 None，optimizer 会跳过），所以不会真正影响 encoder 的权重。但 optimizer 会为这些参数维护动量/方差等状态变量，浪费内存。更重要的是，这暴露了一个理解上的风险——以为 optimizer 会"自动忽略"冻结参数，实际上只是碰巧不会更新而已。

**Q2**：为什么提取嵌入时要用 `torch.no_grad()`？不加会有什么后果？

> 使用 torch.no_grad() 可以告诉 PyTorch 不需要计算梯度，从而节省显存和计算资源。如果不加，PyTorch 会为 encoder 的输出计算梯度，虽然 encoder 的参数被冻结，但这会占用额外的显存，并且可能会导致训练速度变慢。

✅ 正确。补充：不加 `torch.no_grad()` 时，PyTorch 会为 encoder 的每一层构建完整的计算图（用于反向传播），即使参数不更新，这些中间变量也会占用显存。对 469K 参数的模型影响不大，但对大模型来说是致命的。

**Q3**：最终测试集准确率是多少？训练集和测试集差距说明了什么？

> 最终测试集准确率约为 0.9992，训练集 accuracy 在训练过程中逐渐提升，最终达到接近 1.0。训练集和测试集的 accuracy 非常接近，说明模型没有过拟合，能够很好地泛化到未见过的数据。这可能是因为数据生成规则简单，模型容量足够，或者训练步骤足够多。

✅ 准确。这个任务的"泛化"比较简单——数据分布一致（同一种随机采样），不存在分布偏移。真正的泛化挑战需要在不同分布的数据上测试。
