# Week 12 · Day 5 Review：自实现 Encoder 做蛋白质分类

## 一、代码结构分析

### 与任务模板的差异

主人没有照搬 task.md 的骨架代码，而是做了一个**务实的工程决策**：复用 Week 11 的数据模块（`protein_dataset.py` + `esm2_embed.py`），而非自己重新实现 tokenize / collate。

| 项目 | task.md 模板 | 主人实际代码 |
|------|-------------|-------------|
| 数据加载 | `pd.read_csv("data/deeploc_sequences.csv")` | 复用 `load_localization_data()` (HuggingFace datasets) |
| Tokenizer | 自实现字符级 `AA_VOCAB`（23 词表） | 复用 ESM-2 tokenizer（33 词表） |
| Collate | 自实现 `pad_collate` → mask shape `(B,1,1,L)` | 复用 `make_collate_fn` → mask shape `(B,L)` |
| 类别权重 | 无 | ✅ 新增 `compute_class_weights()` |
| 数据划分 | sklearn `train_test_split` | 复用 `split_dataset()` |

**评价**：这是正确的做法。Day 5 的核心目标是「自实现 Encoder 做分类」，数据管线复用已验证的模块，把精力集中在模型训练和结果分析上。比从头写一遍 tokenize 有意义得多。

### 类别权重处理

```python
def compute_class_weights(labels, num_classes):
    weights[c] = total / (num_classes * counts.get(c, 1))
```

这是 inverse frequency weighting，和 Week 11 Day 3 用的方法一致。Nucleus 2424 条 vs Peroxisome 93 条（26 倍差距），不加权重模型会直接放弃少数类。

---

## 二、数据流 / Shape 变化追踪

```
输入序列 (str)
  ↓ ESM-2 tokenizer
input_ids: (B, L)          — L = batch 内最长序列
mask:      (B, L)          — 1=真实 token, 0=padding
  ↓
TransformerEncoder(input_ids, attn_mask)
  attn_mask 处理: mask (B,L) → ~mask.unsqueeze(1).unsqueeze(1).bool() → (B,1,1,L)
  — True=屏蔽, 用于 masked_fill
  ↓
hidden: (B, L, d_model)    — d_model=128
  ↓
Mean pooling:
  m = mask.unsqueeze(-1)   — (B, L, 1)
  hidden = (hidden * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)  — (B, d_model)
  ↓
classifier(hidden)          — (B, num_classes) = (B, 10)
```

### Mask 处理的关键改动

主人在代码注释里记录了：**修改了 `multihead_attention.py` 的 forward，增加了 `attn_mask` 的处理逻辑**。

这是因为 Week 11 的 collate_fn 输出 mask shape 是 `(B,L)`，而 Day 2 的 `MultiHeadAttention` 原本可能不接受外部传入的 attn_mask。主人做了适配，这是合理的工程判断。

同时 `ProteinClassifier.forward` 里做了 mask 的双重处理：
1. **attn_mask**：`~mask.bool()` → 取反，True=屏蔽（PyTorch masked_fill 惯例）
2. **pooling mask**：原始 `mask`，1=真实（用于加权平均）

两套 mask 语义相反，代码里处理得清晰，没有搞混。

---

## 三、关键知识点

### 1. 从头训练 vs 预训练的本质差距

| 模型 | Test Acc | Test Macro F1 | 可训练参数 |
|------|----------|---------------|-----------|
| ESM-2 Frozen | 63.4% | 0.551 | ~42K（分类头） |
| ESM-2 Fine-tuned | 67.7% | 0.581 | ~2.5M（最后2层+头） |
| **自实现 Encoder** | **61.0%** | **0.485** | **600K（全部）** |

自实现模型用了 **60 万参数从头训练**，结果还不如 ESM-2 **4 万参数冻结分类头**。这说明：

- **预训练的价值 >> 模型大小**。ESM-2 在 2.5 亿条蛋白质序列上预训练过，其 Embedding 空间已经编码了氨基酸的生化性质、二级结构倾向、共进化信息。即使完全冻结，这些表示也比从头学 600K 参数强。
- 自实现模型的 600K 参数中，大量容量花在了「学习氨基酸是什么」这种基础知识上，留给「区分亚细胞定位」的容量就不够了。

### 2. 过拟合信号明显

从输出日志看：

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 5 | 1.5970 | 1.5533 | 48.2% | 50.2% |
| 10 | 1.3712 | 1.4897 | 54.4% | 56.1% |
| 15 | 1.2176 | 1.5111 | 58.3% | 57.7% |
| 20 | 1.0151 | 1.5015 | 61.4% | 58.2% |
| 25 | 0.8152 | 1.5993 | 65.6% | 58.1% |
| 30 | 0.7377 | 1.6046 | 67.9% | 57.2% |

- **Val Loss** 从 epoch 10 开始就不再下降（1.49 → 1.60），而 Train Loss 持续下降（1.37 → 0.74）
- **Val Acc** 在 epoch 15-20 达到峰值 ~58%，之后反而下降
- Train 和 Val 之间有 **~10% 的 gap**，典型的过拟合

训练曲线图也印证了这一点：Val Loss 在 epoch 5 之后就翘头了。

### 3. Early Stopping 缺失

代码里没有 Early Stopping 机制。虽然保存了 `best_model.pt`（取 val_acc 最高的 checkpoint），但训练跑了完整 30 epoch。如果加 Early Stopping（patience=5-7），可以在 epoch 15 左右就停，节省训练时间，且 test 结果可能更好（best checkpoint 对应的 val_acc 更高）。

### 4. Best Epoch 分析

从日志看，best_val_acc 出现在 epoch 20 附近（58.19%），但 test_acc 只有 61.0%。等等——test_acc > val_acc？这说明：
- 测试集和验证集的分布可能有微妙差异
- 或者 class_weights 在 evaluate 时也影响了 loss 计算（代码里 `evaluate` 用了带权重的 criterion），但 acc/f1 的计算不依赖权重

---

## 四、踩坑 / 易错点

### 1. Mask 语义翻转

这是最容易出 bug 的地方。代码里有两套 mask：

| 用途 | 变量 | 含义 | 1 表示 |
|------|------|------|--------|
| Attention 内部 | `attn_mask` | 屏蔽 mask | **屏蔽**（True=不要看） |
| Pooling | `mask` | 有效 mask | **有效**（1=真实 token） |

主人处理正确：`attn_mask = ~mask.bool()`。如果搞反了，attention 会关注 padding 而忽略真实 token，模型直接废掉。

### 2. `+ 1e-8` 防除零

```python
hidden = (hidden * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
```

如果某个 batch 里所有序列都是 padding（极端情况），`m.sum()` 为 0，不加 epsilon 会 NaN。虽然实际不太可能发生（序列至少有 50 个 token），但加上是对的。

### 3. ESM-2 tokenizer 加载的额外开销

```python
tokenizer, _ = load_esm2(device='cpu')  # 只要 tokenizer，模型加载后丢弃
```

这会加载完整的 ESM-2 模型权重再丢弃，有几秒的浪费。可以考虑单独只加载 tokenizer（`AutoTokenizer.from_pretrained`），不过影响不大。

---

## 五、输出问题回答

### Q1：梯度裁剪

**梯度裁剪的作用**：限制梯度的最大范数（这里是 L2 范数 ≤ 1.0），防止单步更新过大。

**梯度爆炸的场景**：
- 深层网络中，梯度通过链式法则逐层相乘，如果每层的 Jacobian 矩阵的最大奇异值 > 1，梯度会指数增长
- RNN 中最常见（梯度沿时间步累积）
- Transformer 中较少见（残差连接 + LayerNorm 缓解了），但在训练初期、学习率较大时仍可能发生

主人的回答基本正确。补充一点：`clip_grad_norm_` 是对**所有参数的梯度拼接成一个大向量**做范数裁剪，不是逐参数裁剪。这意味着如果某个参数梯度特别大，它会拉高整体范数，导致其他参数的梯度也被等比缩小。

### Q2：CosineAnnealingLR

主人说"余弦下降，然后周期性回升"——后半句不完全准确。**CosineAnnealingLR 不会自动周期回升**，它只完成一个半余弦周期（从 `lr` 降到 `eta_min`）。如果要周期回升（warm restart），需要用 `CosineAnnealingWarmRestarts`。

在这个任务里，`T_max=30` 意味着学习率从 `1e-3` 沿余弦曲线平滑降到 `1e-5`，总共 30 步。

**优势**：
- 比固定学习率：前期大步探索，后期小步精调
- 比 StepLR：没有突变（阶梯式下降在跳变点可能震荡），平滑过渡

### Q3：差距分析

主人的回答抓住了核心（预训练 + 模型规模），但可以更深入：

**61.0% vs 67.7% 的 6.7% 差距，具体来自：**

1. **预训练表示质量**（最大因素）：ESM-2 在 2.5 亿蛋白质上学习了氨基酸的「语义」——哪些氨基酸功能相似、哪些位置共进化、二级结构倾向。自实现模型只看到 5871 条训练数据，从零学这些。

2. **位置编码质量**：ESM-2 使用可学习的位置编码（Rotary），能更好地捕捉局部 motif 的位置关系。自实现用的是正弦固定编码。

3. **模型容量利用效率**：ESM-2 8M 参数中大部分是预训练好的「通用蛋白质知识」，微调时只需调整 2.5M 就能适配下游任务。自实现的 600K 参数全部从头学，容量不够。

4. **Tokenizer 差异**：虽然都用了 ESM-2 tokenizer（33 词表），但自实现模型的 Embedding 层是从零学的，而 ESM-2 的 Embedding 已经编码了氨基酸的生化性质。

---

## 六、总结

| 维度 | 评价 |
|------|------|
| 代码质量 | ✅ 模块复用合理，mask 处理清晰，无明显 bug |
| 工程决策 | ✅ 复用 Week 11 数据模块、加类别权重、适配 attn_mask——都是正确的判断 |
| 训练结果 | ⚠️ 61.0%，低于 ESM-2 Frozen (63.4%)，验证了预训练的不可替代性 |
| 过拟合 | ⚠️ Val Loss 从 epoch 10 就不再下降，建议下次加 Early Stopping |
| 问题回答 | ✅ Q1/Q3 基本正确，Q2 有小偏差（CosineAnnealing 不会自动回升） |
| 核心收获 | ✅ 用实验数据证明了「预训练 > 从头训练」，这比任何理论解释都有说服力 |

**一句话**：600K 参数从头训练 < 42K 参数冻结预训练——这就是预训练的力量。Day 5 最大的收获不是模型有多好，而是用亲手写的代码验证了这个结论。🐾
