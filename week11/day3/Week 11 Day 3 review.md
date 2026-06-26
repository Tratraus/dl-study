# Week 11 Day 3 Review：冻结 ESM-2 + 分类头训练

## 代码结构分析

### 5 个函数/类的职责

| 函数/类 | 输入 | 输出 | 职责 |
|---------|------|------|------|
| `ProteinClassifier` | (B, 320) | (B, 10) | 两层 MLP：320→128→10 |
| `compute_class_weights` | labels | (10,) 权重 | 处理类别不平衡 |
| `train_one_epoch` | model + data | loss, acc | 训练一个 epoch |
| `evaluate` | model + data | loss, acc | 验证（无梯度） |
| `main()` | — | — | 组装所有步骤 |

### 训练架构

```
蛋白质序列
  │
  ▼ ESM-2（冻结，eval + no_grad）
outputs.last_hidden_state: (B, L+2, 320)
  │
  ▼ mean_pooling_with_mask()
embeddings: (B, 320)
  │
  ▼ ProteinClassifier（可训练）
logits: (B, 10)
  │
  ▼ CrossEntropyLoss(weight=class_weights)
loss → backward → optimizer.step()
```

## 数据流 / Shape 变化追踪

```
DataLoader → (input_ids (B, L), attention_mask (B, L), labels (B,))
  │
  ▼ esm_model(**inputs)
last_hidden_state: (B, L, 320)
  │
  ▼ mean_pooling_with_mask()
(B, 320)
  │
  ▼ classifier
fc1: (B, 320) → (B, 128)
relu: (B, 128) → (B, 128)
dropout: (B, 128) → (B, 128)
fc2: (B, 128) → (B, 10)
  │
  ▼ CrossEntropyLoss
scalar loss
```

## 训练结果分析

### 数值汇总

| 指标 | Epoch 1 | Epoch 5 | Epoch 10 | 趋势 |
|------|---------|---------|----------|------|
| Train Loss | 1.8422 | 1.2002 | 1.0235 | ↓ 持续下降 |
| Train Acc | 46.1% | 61.3% | 65.6% | ↑ 持续上升 |
| Val Loss | 1.5582 | 1.3018 | 1.2573 | ↓ 下降但趋平 |
| Val Acc | 56.7% | 61.5% | 63.2% | ↑ 上升但波动 |

### 关键观察

1. **训练有效**：Val Acc 从 56.7% 提升到 63.4%（最佳），远超随机猜测（10%），说明 ESM-2 的冻结嵌入包含了可区分亚细胞定位的信息。

2. **轻微过拟合**：Epoch 10 时 train acc（65.6%）> val acc（63.2%），差距约 2.4%。不严重，但说明分类器开始记忆训练集。

3. **Val Loss 趋平**：Epoch 8 后 val loss 不再明显下降（1.2489 → 1.2506 → 1.2573），继续训练收益递减。Early Stopping 在 Epoch 8 触发最佳效果。

4. **分类器参数量**：42,378（≈ 320×128 + 128 + 128×10 + 10 = 42,378）。只有 ESM-2 参数量的 0.56%，说明冻结策略下只需要极少可训练参数。

### 类别权重计算验证

```
weight[c] = total / (num_classes × count[c])

Peroxisome: 8388 / (10 × 93)  ≈ 9.02
Nucleus:    8388 / (10 × 2424) ≈ 0.35

权重比：9.02 / 0.35 ≈ 26 倍
```

Peroxisome 的一个样本在 loss 中的贡献是 Nucleus 的 26 倍，防止模型忽略稀有类别。

## 关键知识点

### 1. 冻结 + eval + no_grad 三件套

```python
# 冻结：参数不参与梯度计算
for p in esm_model.parameters():
    p.requires_grad = False

# eval：关闭 Dropout / BatchNorm 的训练行为
esm_model.eval()

# no_grad：不构建计算图，节省显存
with torch.no_grad():
    outputs = esm_model(...)
```

三者缺一不可：
- 只有 `requires_grad=False`：模型仍在 train 模式，Dropout 会随机丢弃
- 只有 `eval()`：参数仍会计算梯度，浪费显存
- 只有 `no_grad()`：Dropout 行为不对

### 2. classifier.train() vs classifier.eval()

| 模式 | Dropout | 用途 |
|------|---------|------|
| `train()` | 随机丢弃 30% 神经元 | 训练时 |
| `eval()` | 不丢弃 | 验证/测试时 |

如果 evaluate 时忘了 `classifier.eval()`，Dropout 仍在工作，验证结果会有随机波动。

### 3. CrossEntropyLoss 内部已含 Softmax

`nn.CrossEntropyLoss` 内部先做 `log_softmax` 再做 `nll_loss`，所以 classifier 的输出是 **logits**（未归一化的分数），不是概率。如果在 forward 里加了 Softmax，loss 会算两次，训练会出问题。

### 4. Adam 优化器只更新 classifier 参数

```python
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
```

虽然 ESM-2 的 `requires_grad=False`，但 optimizer 只注册了 classifier 的参数，所以 ESM-2 的参数不会被更新。两层保险。

## 踩坑 / 易错点

1. **忘记 `classifier.train()`**：如果在 train_one_epoch 里只有 `esm_model.eval()` 而没有 `classifier.train()`，Dropout 不会工作，训练效果可能变差（少了正则化）。

2. **`optimizer.zero_grad()` 的位置**：放在 `loss.backward()` 前面。如果放在 `optimizer.step()` 后面，下一个 batch 的梯度会和上一个累积。

3. **`loss.item()` vs `loss`**：`loss.item()` 返回 Python float，不保留计算图；直接用 `loss` 累积会保持计算图，浪费显存。

## 输出问题回顾

**Q1**: ESM-2 冻结不更新，eval + no_grad 节省计算。如果设为 train()，Dropout 行为会变，且虽然参数不更新但计算图仍会构建，浪费显存。

**Q2**: Peroxisome 权重 ≈ 9.02，Nucleus 权重 ≈ 0.35，比值约 26:1。稀有类别在 loss 中贡献更大，防止模型忽略它们。

**Q3**: 去掉 weight 后模型会倾向预测 Nucleus（占比 28.9%），因为"全猜 Nucleus"就能达到 ~29% 准确率，没有动力学习稀有类别。
